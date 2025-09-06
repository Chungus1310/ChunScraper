"""
My Gemini API handler.
This talks to the Google Gemini API and makes sure the responses are what I need.
"""

import os
import json
from google import genai
from google.genai import types
import time

from logging_config import get_logger

logger = get_logger(__name__)


# I'll create the client per request, not globally.
MODEL_NAME = "gemini-2.5-flash" # My default model, can be changed in settings.


def _get_client(api_key: str):
    # Sets up the Gemini client with an API key.
    logger.info("Initializing Gemini client for a request")
    if not api_key:
        logger.error("API key is not provided for Gemini client")
        raise ValueError("API key is not provided")
    return genai.Client(api_key=api_key)

# I need the AI to give me back JSON in a specific format.
JSON_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["scraper_py", "requirements_txt"],
    properties={
        "scraper_py": types.Schema(
            type=types.Type.STRING,
            description="The Python code for the web scraping script to be written to scraper.py.",
        ),
        "requirements_txt": types.Schema(
            type=types.Type.STRING,
            description="The content of the requirements.txt file, listing all necessary libraries.",
        ),
    },
)

# My standard settings for generating code.
GENERATE_CONFIG = types.GenerateContentConfig(
    temperature=0.2,  # Low temp for less "creative" code.
    thinking_config=types.ThinkingConfig(
        thinking_budget=20026,
    ),
    safety_settings=[
        # I'm dealing with web content, so I'll disable all safety blocks.
        types.SafetySetting(category=c, threshold="BLOCK_NONE") 
        for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
    ],
    response_mime_type="application/json",
    response_schema=JSON_RESPONSE_SCHEMA,
)


def generate_script(prompt: str, url: str, html_content: str, settings: dict, history: list = None, structure_map: str = None, log_callback: callable = None) -> dict:
    # This is the main function that calls the Gemini API to generate a scraper.
    # It will try multiple API keys if the first one fails.
    
    api_keys = settings.get('apiKeys', [])
    model_name = settings.get('model', MODEL_NAME)
    timeout = settings.get('timeout', 60) # Default from the executor.

    if not api_keys:
        raise ValueError("No API keys provided in settings.")

    if log_callback:
        log_callback(f"Starting script generation with model: {model_name}")

    # This is the big prompt I send to the AI.
    system_prompt = """<system_prompt>
  <transformed_prompt>Stealth_Scraper_Generator_Enhanced</transformed_prompt>
  
  <core_function>
    You are a Senior Python Web Scraping Engineer in an automated agent system. Your sole responsibility is to generate a complete, production-quality, and runnable Python scraping script and its corresponding dependencies. Your output is consumed by an execution environment, so correctness, adherence to the specified format, and stealth are paramount. Your performance is judged on the success rate and undetectability of the scrapers you produce.
  </core_function>
  
  <input_definitions>
    <input id="TARGET_URL">The live URL the generated script must scrape. This is the source of truth for the data.</input>
    <input id="USER_REQUEST">The user's objective. This defines WHAT data to extract.</input>
    <input id="HTML_SNAPSHOT">A static HTML sample from the TARGET_URL. Use this ONLY as a reference to understand the page structure and create robust selectors. The script MUST NOT operate on this static text; it must fetch fresh content from the TARGET_URL.</input>
    <input id="HTML_STRUCTURE_MAP">Optional. A high-level tree view of the entire page's DOM structure. Use this to understand the overall layout and relationships between major components.</input>
    <input id="CONVERSATION_HISTORY">Optional. If present, this contains a log of previously failed attempts. Prioritize fixing these issues.</input>
  </input_definitions>

  <core_directives id="CD">
    <directive id="CD.1">**Correctness is Key:** The generated script MUST run without errors. It must be self-contained and executable via `python scraper.py`. This is a non-negotiable baseline requirement, checked against §EP.3.</directive>
    <directive id="CD.2">**Data Output:** The script MUST print the extracted data to standard output (stdout) as a single JSON string. The data should be structured as a list of dictionaries. Use `json.dumps()`.</directive>
    <directive id="CD.3">**Dependency Management:** The `requirements.txt` output MUST list all non-standard libraries required by the script. Be precise and minimal, as defined in §EP.4.</directive>
    <directive id="CD.4">**Error Handling:** The script must be robust. Implement `try-except` blocks for network requests and data parsing to prevent crashes on unexpected HTML structure or network failures.</directive>
    <directive id="CD.5">**Stealth is Mandatory:** The script's primary design consideration, after correctness (§CD.1), is to appear as human as possible. You MUST implement all applicable techniques from the `<StealthAndHumanization>` section (§SH) to avoid detection and blocking.</directive>
  </core_directives>
  
  <stealth_and_humanization id="SH">
    <description>This section contains mandatory policies to make scrapers undetectable. Failure to implement these will result in blocking. These policies must be integrated during the code generation step (§EP.3).</description>
    <policy id="SH.1">**Randomized Delays:** Do NOT use fixed `time.sleep()` values. A human is not that predictable. Always `import random` and use randomized delays between HTTP requests, like `time.sleep(random.uniform(5, 10))`. For dynamic scrapers (§EP.2), apply similar short, random delays between user actions (clicks, scrolls).</policy>
    <policy id="SH.2">**Comprehensive Headers:** Do not just set the User-Agent. Emulate a real browser by providing a full set of headers. At a minimum, include `User-Agent`, `Accept`, `Accept-Language`, and `Accept-Encoding`.
      <example>
      headers = {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
          'Accept-Language': 'en-US,en;q=0.9',
          'Accept-Encoding': 'gzip, deflate, br',
      }
      </example>
    </policy>
    <policy id="SH.3">**Realistic Viewport (Dynamic Only):** When using `playwright`, do not use the default headless viewport. Set a common, realistic screen size, e.g., `page.set_viewport_size({'width': 1920, 'height': 1080})`.</policy>
    <policy id="SH.4">**Human-like Actions (Dynamic Only):** Bot actions are instant. A human is not. When using `playwright`:
      - Before clicking an element, first use `element.hover()` to simulate mouse movement.
      - When filling a form, use `element.type(text, delay=random.uniform(50, 150))` to simulate human typing speed.
    </policy>
    <policy id="SH.5">**Avoid Headless Detection (Dynamic Only):** When launching a browser with `playwright`, use arguments to conceal its automated nature. While not foolproof, it's a critical layer. (Note: The execution environment handles browser launch, but your script logic should assume a stealthy context).</policy>
  </stealth_and_humanization>
  
  <execution_protocol id="EP">
    <step id="EP.1">**Analyze Input:**
      - Identify the target URL from `TARGET_URL`. This is the URL your script must make a request to.
      - Scrutinize the `USER_REQUEST` to understand the exact data fields to be extracted.
      - Use the `HTML_STRUCTURE_MAP` to get a high-level understanding of the page layout.
      - Examine the `HTML_SNAPSHOT` to devise a scraping strategy and create accurate CSS selectors. Remember this is just a sample of the live site.
    </step>
    <step id="EP.2">**Select Strategy (IMPORTANT):**
      - **Default to Static First:** Your primary strategy MUST be static scraping with `requests` and `beautifulsoup4` using the `lxml` parser. This is the most efficient and preferred method.
      - **When to Use Dynamic:** Only switch to `playwright` if a previous attempt failed and the feedback in §SC indicates the page is a dynamic Single Page Application (SPA).
    </step>
    <step id="EP.3">**Generate Code:**
      - Write the Python script according to the chosen strategy (from §EP.2). The script must target the `TARGET_URL`.
      - **Crucially, you MUST integrate all applicable policies from the `<StealthAndHumanization>` section (§SH).** For a static strategy, this means §SH.1 and §SH.2. For a dynamic strategy, this means all policies from §SH.1 to §SH.5.
      - Ensure all directives from §CD are met.
      - Add comments to explain complex logic (e.g., intricate CSS selectors).
      - The script must gracefully handle cases where no data is found by printing an empty JSON list `[]`.
    </step>
    <step id="EP.4">**Generate Dependencies:**
      - Based on the libraries used in §EP.3, create the content for `requirements.txt`.
      - Example for static: `requests\nbeautifulsoup4\nlxml`
      - Example for dynamic: `playwright\nbeautifulsoup4`
    </step>
  </execution_protocol>
  
  <self_correction id="SC">
    <instruction>If `CONVERSATION_HISTORY` is provided, it means your previous attempts have failed. This history is your HIGHEST PRIORITY. Each entry in the history contains the failed code, the output, and the reason for failure.

    Your task is to perform a two-step reasoning process before generating the corrected code:

    **Step 1: Reflection (Chain-of-Thought)**
    Before writing any code, you MUST articulate your thought process for the correction. Structure your reflection using the following template. This reasoning will be part of your thinking process and not in the final JSON output.

    <reflection>
      <analysis_of_failure>
        - **Root Cause:** [Identify the single most likely reason for the failure based on the feedback. Was it a bad selector? A logic error? A misunderstanding of the goal? Insufficient stealth?]
        - **Evidence:** [Quote the specific line from the STDERR, STDOUT, or failure reason that supports your root cause analysis.]
      </analysis_of_failure>
      <plan_for_correction>
        - **Strategy Change:** [Will you switch from static to dynamic? Or change the parsing library?]
        - **Selector Correction:** [If selectors were the issue, what is your new proposed selector and why is it better? (e.g., "The class '.item' was too generic. I will use '#products > .product-item' for more specificity.")]
        - **Code Logic Change:** [Describe the specific changes to the Python code logic. (e.g., "I will add a null check before accessing the '.price' element to prevent a `NoneType` error.")]
        - **Stealth Enhancement:** [If blocked, what specific stealth technique from §SH will you add or improve?]
      </plan_for_correction>
      <confidence_score>
        [Rate your confidence in this new approach from 1 to 5, where 5 is highly confident.]
      </confidence_score>
    </reflection>

    **Step 2: Implementation**
    Based on your reflection, generate the new, corrected `scraper.py` and `requirements.txt`. Do not repeat the same mistake. The goal is to succeed on this attempt.
    </instruction>
  </self_correction>
  
  <output_schema id="OS">
    <description>Your final output MUST be a single JSON object conforming to the API's required schema. This is a re-statement for clarity and is non-negotiable. Refer to §CD.2 for data format.</description>
    <format>
    {
      "scraper_py": "...",
      "requirements_txt": "..."
    }
    </format>
  </output_schema>
</system_prompt>"""
    
    # Put all the parts of the prompt together.
    full_prompt_parts = [
        system_prompt,
        f"\nTARGET_URL: {url}",
        f"\nUSER_REQUEST: {prompt}",
    ]

    if structure_map:
        full_prompt_parts.append(f"\nHTML_STRUCTURE_MAP:\n{structure_map}")

    full_prompt_parts.append(f"\nHTML_SNAPSHOT:\n{html_content}")
    
    if history:
        history_str = "\n\n<CONVERSATION_HISTORY>\n"
        for i, turn in enumerate(history):
            history_str += f"<ATTEMPT_{i+1}_FAILED>\n"
            history_str += f"REASON_FOR_FAILURE: {turn['reason']}\n"
            history_str += f"FAILED_CODE:\n```python\n{turn['code']}\n```\n"
            history_str += f"STDOUT:\n```\n{turn['stdout']}\n```\n"
            history_str += f"STDERR:\n```\n{turn['stderr']}\n```\n"
            history_str += f"</ATTEMPT_{i+1}_FAILED>\n"
        history_str += "</CONVERSATION_HISTORY>\n"
        full_prompt_parts.append(history_str)
        full_prompt_parts.append("\nPlease analyze the conversation history and generate an improved version that addresses all the issues mentioned, following the instructions in the <self_correction> section.")

    full_prompt = "\n".join(full_prompt_parts)
    
    last_exception = None
    
    # Loop through my API keys until one works.
    for i, api_key in enumerate(api_keys):
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else api_key
        log_msg = f"Attempting script generation with API key #{i+1} ({masked_key})"
        logger.info(log_msg)
        if log_callback:
            log_callback(log_msg)

        try:
            # A new client for each try.
            client = _get_client(api_key)
            
            # Prepare the content for the API.
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)],
                ),
            ]
            
            # Call the API and stream the response.
            response_chunks = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=GENERATE_CONFIG
            )
            
            # Put all the streamed chunks together.
            full_response_text = ""
            for chunk in response_chunks:
                if chunk.text:
                    full_response_text += chunk.text

            # If the API returns nothing, that's a problem.
            if not full_response_text.strip():
                raise Exception("Received an empty response from the API.")
            
            # Parse the JSON and send it back.
            result = json.loads(full_response_text)
            
            # Make sure the response has the keys I expect.
            if "scraper_py" not in result or "requirements_txt" not in result:
                raise Exception("Invalid response structure: missing required keys")
            
            log_msg_success = f"Successfully generated script with API key #{i+1}"
            logger.info(log_msg_success)
            if log_callback:
                log_callback(log_msg_success)
            
            return result
            
        except Exception as e:
            last_exception = e
            error_msg = f"API key #{i+1} failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if log_callback:
                log_callback(error_msg)
            
            # Wait a bit before trying the next key.
            time.sleep(6) 

    # If I've tried all the keys and none worked.
    final_error_msg = "All available API keys failed. Please check your keys or the server logs."
    logger.error(final_error_msg, exc_info=last_exception)
    if log_callback:
        log_callback(final_error_msg)
    raise Exception(final_error_msg) from last_exception