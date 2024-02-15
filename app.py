import asyncio
import os
import logging
import re
import shutil
import tempfile
import time
import uuid

import ctranslate2
from transformers import AutoTokenizer

from fastapi import (
    FastAPI,
    Request,
    Response,
    HTTPException,
    status,
    UploadFile,
    Form,
    File,
)
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, field_validator


def check_if_language_is_supported(_lang: str) -> str:
    if os.environ.get("VALIDATE_LANGS", "") == "":
        return _lang
    if _lang in supported_languages:
        return _lang
    else:
        raise HTTPException(
            status_code=400,
            detail=f"{_lang} is not a supported language, see /supported_languages for a list of supported languages!",
        )


# Define the data model for the input and output of the API
class SrcText(BaseModel):
    src_lang: str
    tgt_lang: str
    text: str

    @field_validator("src_lang", "tgt_lang")
    @classmethod
    def check_if_language_is_supported_cls(cls, _lang):
        return check_if_language_is_supported(_lang)


class TgtText(BaseModel):
    text: str


# Initialize the FastAPI app
app = FastAPI()

# Set the maximum upload file size
MAX_UPLOAD_FILE_SIZE = int(os.environ["MAX_UPLOAD_FILE_SIZE"])

# Set up the root logging to follow gunicorn logging settings
# This inplicitly assumes that the API is always run with gunicorn
gunicorn_error_logger = logging.getLogger("gunicorn.error")
logging.root.handlers.extend(gunicorn_error_logger.handlers)

logger = logging.getLogger("MT_API")
logger.setLevel(gunicorn_error_logger.level)

# We assume that the app has access to a NVIDIA GPU
device = "cuda"

logger.info("Loading MT model ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL_STR"])
    translator = ctranslate2.Translator("./_model", device, compute_type="auto")
except Exception as e:
    logger.error(f"Error loading MT model: {repr(e)}")
    raise e
else:
    logger.info("MT model loaded!")

supported_languages = []
if os.environ.get("VALIDATE_LANGS", "") !="":
    logger.info("Loading supported languages ...")
    try:
        with open("./supported_languages.txt") as f:
            supported_languages = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        logger.warning(
            "supported_languages.txt not found, disabling source/target language validation!"
        )
        os.environ["VALIDATE_LANGS"] = ""
    else:
        if supported_languages == []:
            logger.warning(
                "supported_languages.txt is empty, disabling source/target language validation!"
            )
            os.environ["VALIDATE_LANGS"] = ""
        else:
            logger.info(
                "supported_languages.txt found, enabling source/target language validation!"
            )
            logger.info("Supported langauges loaded!")


if os.environ.get("SPLIT_SENTENCES", "") != "":
    logger.info("Loading sentence splitter ...")
    from wtpsplit import WtP

    try:
        wtp = WtP("wtp-bert-mini")
        wtp.half().to("cuda")
    except Exception as e:
        logger.error(f"Error loading sentence splitter: {repr(e)}")
        raise e
    else:
        logger.info("Sentence splitter loaded!")


# Define a middleware to log all requests
@app.middleware("http")
async def log_requests(request, call_next):
    ruuid = str(uuid.uuid4())
    logger.info(f"RID={ruuid} REQUEST_PATH={request.url.path}")
    request.state.ruuid = ruuid

    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    formatted_process_time = "{0:.2f}".format(process_time)
    logger.info(
        f"RID={ruuid} COMPLETED_IN={formatted_process_time}ms STATUS_CODE={response.status_code}"
    )

    return response


# Define the translation function
async def translate_func(src: SrcText, ruuid: str):
    try:
        tokenizer._src_lang = src.src_lang
        tokenizer.set_src_lang_special_tokens(src.src_lang)

        logger.debug(f"RID={ruuid} INPUT_STR --- {src.text}")
        _input = tokenizer.convert_ids_to_tokens(tokenizer.encode(src.text))
        _input_rep = " ".join(_input)
        logger.debug(f"RID={ruuid} TOKENIZED_INPUT_STR --- {_input_rep}")

        out = translator.translate_batch(
            [_input],
            target_prefix=[[src.tgt_lang]],
            beam_size=1,
            max_input_length=512,
            max_decoding_length=1024,
            max_batch_size=1,
            batch_type="examples",
        )
        out = out[0].hypotheses[0][1:]
        _out_rep = " ".join(out)
        logger.debug(f"RID={ruuid} TOKENIZED_OUTPUT_STR --- {_out_rep}")
        out_txt = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(out), skip_special_tokens=True
        )
        logger.debug(f"RID={ruuid} OUTPUT_STR --- {out_txt}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error translating text: {repr(e)}"
        )

    return out_txt


# Define the API translation endpoint
@app.post("/translate", response_model=TgtText)
async def translate_text(src: SrcText, request: Request):
    ruuid = request.state.ruuid

    if src.text == "":
        return {"text": ""}

    if re.fullmatch("^\s+$", src.text):
        return {"text": src.text}

    if os.environ.get("SPLIT_SENTENCES", "") != "":
        _out_txt = await asyncio.gather(
            *[
                translate_func(
                    SrcText(
                        src_lang=src.src_lang, tgt_lang=src.tgt_lang, text=sent.strip()
                    ),
                    ruuid,
                )
                for sent in wtp.split(src.text)
            ]
        )
        _out_txt = " ".join(_out_txt)
    else:
        _out_txt = await translate_func(src, ruuid)
    return {"text": _out_txt}


@app.post("/translate_file")
async def translate_file(
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    file: UploadFile = File(...),
    request: Request = None,
):
    ruuid = request.state.ruuid

    # Will throw an error if the source/target language is not supported
    _ = check_if_language_is_supported(src_lang)
    _ = check_if_language_is_supported(tgt_lang)

    # Will throw an error if the file is not a .txt file
    file_name, file_extension = os.path.splitext(file.filename)
    _mt_file_name = f"{file_name}.{tgt_lang}.txt"
    if file_extension != ".txt":
        raise HTTPException(
            status_code=400, detail="Uploaded file must be a .txt file!"
        )

    # Read the file, throwing an error if it's too large
    _file_size = 0
    _file_body = b""
    for chunk in file.file:
        _file_size += len(chunk)
        if _file_size > MAX_UPLOAD_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"The uploaded file is too large, max size is {MAX_UPLOAD_FILE_SIZE} bytes!",
            )
        _file_body += chunk

    try:
        decoded_text = _file_body.decode("utf-8").strip()
        tmpdirname = tempfile.mkdtemp()

        with open(os.path.join(tmpdirname, _mt_file_name), "w") as f:
            for _line in decoded_text.split("\n"):
                _mt_text = await translate_text(
                    SrcText(src_lang=src_lang, tgt_lang=tgt_lang, text=_line.strip()),
                    request,
                )
                f.write(_mt_text["text"] + "\n")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error translating file: {repr(e)}"
        )

    return FileResponse(
        os.path.join(tmpdirname, _mt_file_name),
        media_type="text/plain",
        filename=_mt_file_name,
        background=BackgroundTask(shutil.rmtree, tmpdirname),
    )


# Return the list of supported languages
@app.get("/supported_languages", status_code=200)
async def get_supported_languages(response: Response):
    if supported_languages == []:
        logger.warning(
            "supported_languages.txt is empty or source/target language validation is disabled!"
        )
        response.status_code = status.HTTP_204_NO_CONTENT
    else:
        return {"supported_languages": supported_languages}
