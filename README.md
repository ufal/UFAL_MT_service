# UFAL Multilingual Translation Service

This text provides details on the deployment and usage of the Machine Translaton service developed by UFAL for the purpose of the EC HE MEMORISE project.


## Deployment

The service is implemented as a single Docker container and it sets up a RESTful API. The necessary artifacts (models) are downloaded from the web during the building stage.


### Requirements

The servis requires a NVIDIA GPU with [Compute Capability >= 7.5](https://developer.nvidia.com/cuda-gpus) and at least 8GB of VRAM. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) may be required for the Docker to be able to access the GPU. Additionaly, roughly 35 GB of disc space are required for the intermediate representations during the building stage. At the runtime, we recommend at least 8GB of RAM and at least 8 CPU threads.


### Image Specification

Building the container:

``` bash
docker build -t ufal-mt-service:1.0.0 .
```
Parameters (and their default values) available at the building stage (via the `--build-arg=`):
* `HF_MODEL_STR="facebook/nllb-200-3.3B"` - the MT engine to be used inside the API. The string must correspond to a valid [HuggingFace/Transformers](https://huggingface.co/docs/transformers/en/index) model that is also supported by the [CTranslate2](https://opennmt.net/CTranslate2/guides/transformers.html) framework.
* `VALIDATE_LANGS="True"` - whether to validate the source/target languages provided in the request. To turn this off, set the variable to en empty string, i.e., `""`. To work properly, a `.txt` file with supported languages, one language per line, should be provided, see `supported_languages.txt`.
* `SPLIT_SENTENCES="True"` - whether to apply a [sentence splitter](https://github.com/bminixhofer/wtpsplit) to the input text. To turn this off, set the variable to en empty string, i.e., `""`. Most of the MT engines expect their input to consist of a single sentence, thus this pre-processing step.
* `LOG_LEVEL="info"` - the logging level to be used. For debugging purposes, run with `LOG_LEVEL="debug"`.
* `MAX_UPLOAD_FILE_SIZE="10485760"` - we limit the maximum file size that can be uploaded with the `/translate_file` endpoint. The default value corresponds to 10 MB (10485760 = 10 * 1024 * 1024) and should suffice for a file with roughly 50,000 lines of text, with a single sentence per line.

The building stage should take roughly 15-20 minutes.

Running the container:
``` bash
docker run \
-p {PORT_TO_BIND}:8081 \
-v {LOCAL_FOLDER}:/mt_logs \
--gpus {GPU_IDENTIFIER} \
ufal-mt-service:1.0.0
```
Most of the parameters from the building stage (`VALIDATE_LANGS`, `SPLIT_SENTENCES`, `LOG_LEVEL`, `MAX_UPLOAD_FILE_SIZE`) can also be modified (`-e`) at the runtime , e.g., `docker run -e LOG_LEVEL="debug"`.

The runtime specific parameters:

* `PORT_TO_BIND` - the local TCP port on `127.0.0.1`/`localhost` of the host machine used to query the translation service running inside the container.
* `LOCAL_FOLDER` - a folder on the local host machine where the API logs will be stored. If started successfully, a single file (`mt.log`) will be created.
* `GPU_IDENTIFIER` - a valid GPU identifier, specific to the host machine. The API is capable of using only a single GPU, thus `--gpus 'device=0'` should suffice in most cases, when run on a machine with a single GPU.

The startup should take no more than 1 minute. Once the `[INFO] [uvicorn.error] [Application startup complete.]` message appears in the logs, the service is operational.


## API Endpoints

The service provides three endpoints:


### `POST /translate`

Given the source language, allows translating the provided text into the target language.
Content type: `application/json`.

<h3 id="post__translate-parameters">Parameters</h3>

|Name|Type|Required|Notes|
|---|---|---|---|
|tgt_lang|enum/string|true| a [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) code of the language you want to translate the text into. Use the `/supported_languages` endpoint to obtain the list of currently supported values. |
|src_lang|enum/string|true| a [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) code indicating the source language of your text. Use the `/supported_languages` endpoint to obtain the list of currently supported values. |
|text|string|true| Input text. If the `SPLIT_SENTENCES` option is activated, a sentence-splitter will be applied to the text. Each sentence will be translated independently, and the translations will be merged prior to returning the outcome. There is no guarantee that new lines (`\n`) will be preserved - in order to preserve multi-line structure use the `/translate_file` endpoint. Each request (or each sentence, if using sentence-splitter) will be truncated to 512 [tokens](https://huggingface.co/docs/transformers/main_classes/tokenizer) (roughly 150 words).|

<h3 id="post__translate-responses">Response Codes</h3>

|Status|Meaning|Description|
|---|---|---|
|200|OK|Success.
|400/422|Bad Request|Validation error.
|500|Internal Server Error|Failed to process the request because of an internal problem.

<h3 id="post__translate-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Value|
|---|---|---|
|text|string|Translation output|

Status Code **400**

|Name|Type|Value|
|---|---|---|
|detail|json|Brief description of the validation error.|

Status Code **422**

|Name|Type|Value|
|---|---|---|
|detail|json|Brief description of the validation error.|

<h3 id="post__translate-examples">Examples</h3>

Request:

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{
        "src_lang": "eng_Latn",
        "tgt_lang": "ces_Latn",
        "text": "How are you today?"
        }'
```

Response (200):

```json
{
    "text":"Jak se dnes máte?"
}
```

Request (missing `tgt_lang`):

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{
        "src_lang": "deu_Latn",
        "text": "Ich möchte schlafen."
        }'
```

Response (422):

```json
{
    "detail":[{
        "type":"missing",
        "loc":["body","tgt_lang"],
        "msg":"Field required",
        "input":{
            "src_lang":"deu_Latn",
            "text":"Ich möchte schlafen."
            },
        "url":"https://errors.pydantic.dev/2.6/v/missing"
        }]
}
```
Request (invalid language code):

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{
        "src_lang": "pol_Latn",
        "tgt_lang": "English",
        "text": "Mam nadzieję, że następnym razem ci się uda!"
        }'
```

Response (400):

```json
{
    "detail":"English is not a supported language, see /supported_languages for a list of supported languages!"
}
```

### `POST /translate_file`

Given the source language, allows translating the provided text file into the target language.
Content type: `multipart/form-data`.

<h3 id="post__translate-parameters">Parameters</h3>

|Name|Type|Required|Notes|
|---|---|---|---|
|tgt_lang|enum/string|true| a [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) code of the language you want to translate the text file into. Use the `/supported_languages` endpoint to obtain the list of currently supported values. |
|src_lang|enum/string|true| a [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) code indicating the source language of your text file. Use the `/supported_languages` endpoint to obtain the list of currently supported values. |
|file|local path/string|true| The local text file that one wants to translate. The required format is `.txt`. Each line will be translated independently, preserving line ordering. All lines must be in the same language, `src_lang`. If the `SPLIT_SENTENCES` option is activated, a sentence-splitter will be applied to each line. Each sentence will be translated independently, and the translations will be merged prior to returning the outcome. Each line (or each sentence, if using sentence-splitter) will be truncated to 512 [tokens](https://huggingface.co/docs/transformers/main_classes/tokenizer) (roughly 150 words).|


<h3 id="post__translate-responses">Response Codes</h3>

|Status|Meaning|Description|
|---|---|---|
|200|OK|Success.
|400/422|Bad Request|Validation error.
|500|Internal Server Error|Failed to process the request because of an internal problem.

<h3 id="post__translate-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Value|
|---|---|---|
|-|text/plain; charset=utf-8|The translated text|

<h4 id="post__translate-responseschema">Details</h4>

We recommend using one of the following options for handling the output (using `CURL` terminology):
* `-O -J` - when translating `LOCAL_FILE.txt` into `tgt_lang`, the default output file name is `LOCAL_FILE.tgt_lang.txt` 
* `-o OUTPUT_NAME.txt` - can be used to specify a target file name
* ` ` - if neither of those options is provided, the outcome will be a plain text

Status Code **400**

|Name|Type|Value|
|---|---|---|
|detail|string|Brief description of the validation error.|

Status Code **422**

|Name|Type|Value|
|---|---|---|
|detail|json|Brief description of the validation error.|

<h3 id="post__translate-examples">Examples</h3>

Request:

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate_file' \
    -F src_lang=eng_Latn \
    -F tgt_lang=arb_Arab \
    -F file=@LOCAL_FILE.txt \
    -O -J
```

Response (200):

```bash
curl: Saved to filename 'LOCAL_FILE.arb_Arab.txt'
```

Request:

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate_file' \
    -F src_lang=eng_Latn \
    -F tgt_lang=fra_Latn \
    -F file=@LOCAL_FILE.txt
```

Response (200):

```text
Lohan a crié Kanye West et remercié Le président Obama pour avoir inspiré la candidature de 2020
Moscou a déclaré que ses dernières frappes en Syrie ont touché 49 cibles, dont un camp d'entraînement de kamikazes.
La clôture est prévue mardi, mais le juge ne devrait pas
Une conversation avec le résistant résurrectionniste de Game of Thrones Melisandre (alias l'actrice Carice Van Houten) sur les sentiments chaleureux et flou de sa mère pour l'acteur Le père de Kit Harington
Trois personnes ont été tuées et une autre hospitalisée après un accident entre deux véhicules au nord de Perth.
```

Request (missing `src_lang`):

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate_file' \
    -F tgt_lang=fra_Latn \
    -F file=@LOCAL_FILE.txt
```

Response (422):

```json
{
    "detail":[{
        "type":"missing",
        "loc":["body","src_lang"],
        "msg":"Field required",
        "input": null
        "url":"https://errors.pydantic.dev/2.6/v/missing"
        }]
}
```

Request (invalid language code):

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate_file' \
    -F src_lang=eng_Latn \
    -F tgt_lang=French \
    -F file=@LOCAL_FILE.txt
```

Response (400):

```json
{
    "detail":"French is not a supported language, see /supported_languages for a list of supported languages!"
}
```

Request (invalid file format):

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate_file' \
    -F src_lang=eng_Latn \
    -F tgt_lang=fra_Latn \
    -F file=@LOCAL_FILE.json
```

Response (400):

```json
{
    "detail":"Uploaded file must be a .txt file!"
}
```

Request (file too large):

```bash
curl -X 'POST' 'http://localhost:PORT_TO_BIND/translate_file' \
    -F src_lang=eng_Latn \
    -F tgt_lang=fra_Latn \
    -F file=@HUGE_LOCAL_FILE.txt
```

Response (400):

```json
{
    "detail":"The uploaded file is too large, max size is 10485760 bytes!"
}
```


### `GET /supported_languages`

Returns the list of supported source/target languages.

<h3 id="get__status-responses">Response Codes</h3>

|Status|Meaning|Description|
|---|---|---|
|200|OK|Success.|
|204|No Content|Language validation is disabled or `supported_languages.txt` couldn't be properly parsed - the response is empty.
|500|Internal Server Error|Failed to process the request because of an internal problem.|

<h3 id="get__status-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Value|
|---|---|---|
|supported languages|array of strings|Supported languages|

<h3 id="get__status-examples">Examples</h3>

Request:

```bash
curl -X 'GET' 'http://localhost:PORT_TO_BIND/supported_languages'
```

Response (200):

```json
{
    "supported_languages":["ace_Arab","ace_Latn","acm_Arab", ..., "zho_Hans","zho_Hant","zul_Latn"]
}
```

Request (with `VALIDATE_LANGS=''`):

```bash
curl -X 'GET' 'http://localhost:PORT_TO_BIND/supported_languages'
```

Response (204):

```bash
```


## Logging

Currently, the API is logging into a .log file, as described in `Image Specification`. The logging configuration is specified via the `gunicorn_logging.conf` file. We recommend using the `-v` option to map the logging file to a local one. By default, logging level is set to `info`. For more detailed information, change it to `debug`. Logger follows the formatting described below:

```bash
[TIME IN %Y-%m-%d %H:%M:%S FORMAT] [LOGGING LEVEL] [LOGGER NAME] [MESSAGE]
```

The API-specif logger formats the MESSAGE with an additional ID (`RID=`), that can be used to filter all information related to a particular request.


### Example (info)

``` bash
[2024-02-12 08:53:14] [INFO] [MT_API] [Loading MT model ...]
[2024-02-12 08:53:27] [INFO] [MT_API] [Sentence splitter loaded!]
[2024-02-12 08:53:27] [INFO] [uvicorn.error] [Application startup complete.]
[2024-02-12 09:06:12] [INFO] [MT_API] [RID=693df365-ddbd-459a-96dd-91dd1c5c9bb3 REQUEST_PATH=/translate_file]
[2024-02-12 09:06:48] [INFO] [MT_API] [RID=693df365-ddbd-459a-96dd-91dd1c5c9bb3 COMPLETED_IN=35713.60ms STATUS_CODE=200]                                                               
```

### Example (debug)

```bash
[2024-02-12 12:31:23] [INFO] [MT_API] [RID=dc2fbfde-4990-4599-9d3a-6eeaf1f13220 REQUEST_PATH=/translate]
[2024-02-12 12:31:24] [DEBUG] [MT_API] [RID=dc2fbfde-4990-4599-9d3a-6eeaf1f13220 INPUT_STR --- How are you today?]
[2024-02-12 12:31:24] [DEBUG] [MT_API] [RID=dc2fbfde-4990-4599-9d3a-6eeaf1f13220 TOKENIZED_INPUT_STR --- eng_Latn ▁How ▁are ▁you ▁today ? </s>]
[2024-02-12 12:31:24] [DEBUG] [MT_API] [RID=dc2fbfde-4990-4599-9d3a-6eeaf1f13220 TOKENIZED_OUTPUT_STR --- ▁Jak ▁se ▁dnes ▁máte ?]
[2024-02-12 12:31:24] [DEBUG] [MT_API] [RID=dc2fbfde-4990-4599-9d3a-6eeaf1f13220 OUTPUT_STR --- Jak se dnes máte?]
[2024-02-12 12:31:24] [INFO] [MT_API] [RID=dc2fbfde-4990-4599-9d3a-6eeaf1f13220 COMPLETED_IN=645.02ms STATUS_CODE=200]
[2024-02-12 12:31:24] [INFO] [uvicorn.access] [172.17.0.1:36610 - POST /translate HTTP/1.1 200]
```


## Latency tests

In the table below you can find estimates of the translation times, using the `translate_file` endpoint with sentence splitter enabled. The GPU that was used for testing is RTX 2080 with 11GB of VRAM and the MT engine is the default one, i.e., `facebook/nllb-200-3.3B`. We measured total time required to translate 100 sentences, using three different splits:
* **short sentences** - sentences no longer than 10 words
* **medium sentences** - sentences with more than 10 words and less than 50 words
* **long sentences** - sentences with more than 50 words and less than 150 words

The reported results are for translations from English into German, averaged over 10 runs (each run with different 100 sentences, sampled from each bucket type).


||**short sentences**|**medium sentences**|**long sentences**|
|---|---|---|---|
|Average number of words | 7.4 | 23.9 | 72.6 | 
|Average number of characters | 49.5 | 146.0 | 435.9 | 
|Average translation time for 100 sentences| 19.6 ± 2.0 | 38.8 ± 0.74 | 104.4 ±  3.9 | 
