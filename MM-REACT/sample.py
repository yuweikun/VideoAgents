### Make sure to set up environment variables first

import os
import json
import requests
from pathlib import Path
from typing import Any, Dict

from langchain.agents import initialize_agent, Tool
from langchain.tools.bing_search.tool import BingSearchRun, BingSearchAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import PALChain
from langchain.llms import AzureOpenAI
from langchain.utilities import ImunAPIWrapper, ImunMultiAPIWrapper
from langchain.utils import get_url_path

# Make repo root importable for driver eval template/payload
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.append(str(ROOT))

try:
    from driver_eval_qwen_mmreact import build_payload, run_once
except Exception:
    build_payload = None
    run_once = None

MAX_TOKENS = 512

# This is private endpoint, will have to change to turbo later
llm = AzureOpenAI(deployment_name="gpt-35-turbo-version-0301", model_name="gpt-35-turbo (version 0301)", temperature=0, max_tokens=MAX_TOKENS)

memory = ConversationBufferMemory(memory_key="chat_history")

imun_dense = ImunAPIWrapper(
    imun_url=os.environ["IMUN_URL2"],
    params=os.environ["IMUN_PARAMS2"],
    imun_subscription_key=os.environ["IMUN_SUBSCRIPTION_KEY2"])

imun = ImunAPIWrapper()
imun = ImunMultiAPIWrapper(imuns=[imun, imun_dense])

imun_celeb = ImunAPIWrapper(
    imun_url=os.environ["IMUN_CELEB_URL"],
    params="")

imun_read = ImunAPIWrapper(
    imun_url=os.environ["IMUN_OCR_READ_URL"],
    params=os.environ["IMUN_OCR_PARAMS"],
    imun_subscription_key=os.environ["IMUN_OCR_SUBSCRIPTION_KEY"])

imun_receipt = ImunAPIWrapper(
    imun_url=os.environ["IMUN_OCR_RECEIPT_URL"],
    params=os.environ["IMUN_OCR_PARAMS"],
    imun_subscription_key=os.environ["IMUN_OCR_SUBSCRIPTION_KEY"])

imun_businesscard = ImunAPIWrapper(
    imun_url=os.environ["IMUN_OCR_BC_URL"],
    params=os.environ["IMUN_OCR_PARAMS"],
    imun_subscription_key=os.environ["IMUN_OCR_SUBSCRIPTION_KEY"])

imun_layout = ImunAPIWrapper(
    imun_url=os.environ["IMUN_OCR_LAYOUT_URL"],
    params=os.environ["IMUN_OCR_PARAMS"],
    imun_subscription_key=os.environ["IMUN_OCR_SUBSCRIPTION_KEY"])

imun_invoice = ImunAPIWrapper(
    imun_url=os.environ["IMUN_OCR_INVOICE_URL"],
    params=os.environ["IMUN_OCR_PARAMS"],
    imun_subscription_key=os.environ["IMUN_OCR_SUBSCRIPTION_KEY"])

bing = BingSearchAPIWrapper(k=2)

def edit_photo(query: str) -> str:
    endpoint = os.environ["PHOTO_EDIT_ENDPOINT_URL"]
    query = query.strip()
    url_idx, img_url = get_url_path(query)
    if not img_url.startswith(("http://", "https://")):
        return "Invalid image URL"
    img_url = img_url.replace("0.0.0.0", os.environ["PHOTO_EDIT_ENDPOINT_URL_SHORT"])
    instruction = query[:url_idx]
    # This should be some internal IP to wherever the server runs
    job = {"image_path": img_url, "instruction": instruction}
    response = requests.post(endpoint, json=job)
    if response.status_code != 200:
        return "Could not finish the task try again later!"
    return "Here is the edited image " + endpoint + response.json()["edited_image"]


def driver_eval_tool(query: str) -> str:
    """
    Run driver evaluation using Qwen3-VL. Expects JSON string with:
    {
      "folder": "01-1",
      "file_stem": "start_at_min02sec03",
      "video_dir": "optional, defaults to sample_data/<folder>"
    }
    """
    if build_payload is None or run_once is None:
        return "driver_eval_qwen_mmreact is not available."
    try:
        params: Dict[str, Any] = json.loads(query)
    except Exception:
        return "Invalid input. Please pass JSON with folder/file_stem."
    folder = params.get("folder", "01-1")
    file_stem = params.get("file_stem", "start_at_min02sec03")
    video_dir = params.get("video_dir", str(ROOT / "sample_data" / folder))
    video_path = Path(video_dir) / f"{file_stem}.mp4"
    if not video_path.exists():
        return f"Video not found: {video_path}"
    payload = build_payload(folder, file_stem)
    result = run_once(video_path, payload)
    return result.get("draft_report", "")

# these tools should not step on each other's toes
tools = [
    Tool(
        name="PAL-MATH",
        func=PALChain.from_math_prompt(llm).run,
        description=(
        "A wrapper around calculator. "
        "A language model that is really good at solving complex word math problems."
        "Input should be a fully worded hard word math problem."
        )
    ),
    Tool(
        name = "Image Understanding",
        func=imun.run,
        description=(
        "A wrapper around Image Understanding. "
        "Useful for when you need to understand what is inside an image (objects, texts, people)."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    Tool(
        name = "OCR Understanding",
        func=imun_read.run,
        description=(
        "A wrapper around OCR Understanding (Optical Character Recognition). "
        "Useful after Image Understanding tool has found text or handwriting is present in the image tags."
        "This tool can find the actual text, written name, or product name in the image."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    Tool(
        name = "Receipt Understanding",
        func=imun_receipt.run,
        description=(
        "A wrapper receipt understanding. "
        "Useful after Image Understanding tool has recognized a receipt in the image tags."
        "This tool can find the actual receipt text, prices and detailed items."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    Tool(
        name = "Business Card Understanding",
        func=imun_businesscard.run,
        description=(
        "A wrapper around business card understanding. "
        "Useful after Image Understanding tool has recognized businesscard in the image tags."
        "This tool can find the actual business card text, name, address, email, website on the card."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    Tool(
        name = "Layout Understanding",
        func=imun_layout.run,
        description=(
        "A wrapper around layout and table understanding. "
        "Useful after Image Understanding tool has recognized businesscard in the image tags."
        "This tool can find the actual business card text, name, address, email, website on the card."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    Tool(
        name = "Invoice Understanding",
        func=imun_invoice.run,
        description=(
        "A wrapper around invoice understanding. "
        "Useful after Image Understanding tool has recognized businesscard in the image tags."
        "This tool can find the actual business card text, name, address, email, website on the card."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    Tool(
        name = "Celebrity Understanding",
        func=imun_celeb.run,
        description=(
        "A wrapper around celebrity understanding. "
        "Useful after Image Understanding tool has recognized people in the image tags that could be celebrities."
        "This tool can find the name of celebrities in the image."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    BingSearchRun(api_wrapper=bing),
    Tool(
        name = "Photo Editing",
        func=edit_photo,
        description=(
        "A wrapper around photo editing. "
        "Useful to edit an image with a given instruction."
        "Input should be an image url, or path to an image file (e.g. .jpg, .png)."
        )
    ),
    Tool(
        name="Driver Evaluation (Qwen3-VL)",
        func=driver_eval_tool,
        description=(
            "Generate a 5-part driver co-pilot evaluation report (scene, attention, HMI, evaluation, recommended actions). "
            "Input: JSON string with keys folder (e.g., '01-1'), file_stem (e.g., 'start_at_min02sec03'), "
            "optional video_dir (defaults to sample_data/<folder>)."
        ),
    ),
]

chain = initialize_agent(tools, llm, agent="conversational-mm-assistant", verbose=True, memory=memory, return_intermediate_steps=True, max_iterations=4)
output = chain.conversation("https://www.oracle-dba-online.com/sql/weekly_sales_table.png")
