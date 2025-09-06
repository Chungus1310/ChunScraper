"""
The brains of the operation.
This file orchestrates the whole process of generating, testing, and fixing the scrapers.
"""

import os
import tempfile
import shutil
import zipfile
import uuid
import time
import json
import re
from typing import Dict, Callable, Optional, Any
import httpx
from bs4 import BeautifulSoup

from logging_config import get_logger
from gemini_service import generate_script
from executor import run_script_in_dir

logger = get_logger(__name__)


def create_html_structure_map(html_content: str, log_callback: Optional[Callable[[str], None]] = None) -> str:
    # I'm creating a simplified map of the HTML to help the AI understand the page layout.
    # It's like a table of contents for the webpage.
    def log(message):
        logger.info(message)
        if log_callback:
            log_callback(message)
            
    try:
        log("Generating HTML structure map...")
        soup = BeautifulSoup(html_content, 'html.parser')
        body = soup.find('body')
        if not body:
            log("Could not find <body> tag to generate structure map.")
            return "<body> tag not found."

        # A little recursive function to build the tree.
        def build_tree(element, indent="", depth=0):
            if not hasattr(element, 'name') or not element.name:
                return ""

            # I'll grab the tag name, id, and a few classes.
            tag_info = element.name
            if element.has_attr('id'):
                tag_info += f"#{element['id']}"
            if element.has_attr('class'):
                tag_info += f".{'.'.join(element.get('class', [])[:3])}"

            tree_str = f"{indent}<{tag_info}>\n"
            
            # I'll limit how deep and wide this map goes to keep it readable.
            if depth > 7:
                return tree_str + f"{indent}  [...max depth reached...]\n"

            child_count = 0
            for child in element.find_all(recursive=False):
                if child.name:  # Only care about tags.
                    if child_count > 10:  # Not too many children per node.
                        tree_str += f"{indent}  [...and more...]\n"
                        break
                    tree_str += build_tree(child, indent + "  ", depth + 1)
                    child_count += 1
            return tree_str

        map_str = build_tree(body)
        log(f"Successfully generated HTML structure map of length {len(map_str)}.")
        return map_str
        
    except Exception as e:
        log(f"Could not generate HTML structure map: {e}")
        return "Error: Could not generate HTML structure map."


def _expand_html_context(full_html: str, last_snippet: str, log_callback: Optional[Callable[[str], None]] = None) -> str:
    # If a scrape fails, I'll try to give the AI a bit more context from the HTML.
    # This finds the parent of the last piece of HTML it looked at.
    def log(message):
        logger.info(message)
        if log_callback:
            log_callback(message)

    try:
        full_soup = BeautifulSoup(full_html, 'html.parser')
        snippet_soup = BeautifulSoup(last_snippet, 'html.parser')
        
        # Find the first real element in the snippet.
        first_element_in_snippet = snippet_soup.find(lambda tag: tag.name != 'html' and tag.name != 'body')

        if not first_element_in_snippet:
            log("Could not find a significant element in the last snippet. Returning full body.")
            body = full_soup.find('body')
            return str(body) if body else full_html

        # Try to find the same element in the full HTML by looking for its text.
        text_content = first_element_in_snippet.get_text(" ", strip=True)
        if not text_content:
             log("First element in snippet has no text content. Returning full body.")
             body = full_soup.find('body')
             return str(body) if body else full_html

        # Use a regex to be a bit fuzzy with the text matching.
        regex = re.compile(r'\s*'.join(re.escape(word) for word in text_content.split()[:15])) # Match the first 15 words.
        elements_in_full_doc = full_soup.find_all(text=regex)

        if not elements_in_full_doc:
            log("Could not locate the snippet's content in the full HTML. Returning full body.")
            body = full_soup.find('body')
            return str(body) if body else full_html

        # Get the parent of the element I found.
        target_element = elements_in_full_doc[0].find_parent()

        if target_element and target_element.parent and target_element.parent.name not in ['body', 'html']:
            parent_html = str(target_element.parent)
            log(f"Successfully found parent element <{target_element.parent.name}>. New context length: {len(parent_html)}")
            return parent_html
        else:
            log("Parent is body, html, or not found. Returning the full body as context.")
            body = full_soup.find('body')
            return str(body) if body else full_html

    except Exception as e:
        log(f"Error during HTML context expansion: {e}. Falling back to full body content.")
        return full_html


def extract_relevant_html(html_content: str, user_prompt: str) -> str:
    # I'm trying to be smart here and only send the most relevant parts of the HTML to the AI.
    # This should save tokens and improve accuracy.
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        prompt = user_prompt.lower()

        # First, I'll get rid of stuff that's usually not useful for scraping.
        for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
            element.decompose()

        # --- Stage 1: Try to find the main content area of the page. ---
        main_content_soup = None
        main_selectors = ['main', '[role="main"]', 'article', '#content', '#main', '.content', '.main']
        for selector in main_selectors:
            main_content_area = soup.select_one(selector)
            if main_content_area:
                main_content_soup = main_content_area
                logger.info(f"Found main content area with selector: '{selector}'")
                break
        
        search_scope = main_content_soup if main_content_soup else soup.find('body') or soup

        # --- Stage 2: Look for keywords from the user's prompt in the HTML. ---
        structural_keywords = {
            # Layout stuff
            'card': ['[class*="card"]'], 'container': ['[class*="container"]'], 'wrapper': ['[class*="wrapper"]'],
            'section': ['section'], 'block': ['[class*="block"]'], 'grid': ['[class*="grid"]'],
            'list': ['ul', 'ol'], 'item': ['li', '[class*="item"]'],
            'row': ['tr', '[class*="row"]'], 'column': ['td', 'th', '[class*="col"]'],
            # Content stuff
            'table': ['table'],
            'article': ['article'], 'post': ['[class*="post"]'], 'comment': ['[class*="comment"]'], 'review': ['[class*="review"]', '[itemtype*="Review"]'],
            'title': ['[class*="title"]', '.title'], 'header': ['h1', 'h2', 'h3', '[class*="header"]'], 'headline': ['h1', 'h2', 'h3'],
            'description': ['[class*="description"]', '[class*="desc"]'], 'summary': ['[class*="summary"]'],
            'author': ['[class*="author"]', '[rel="author"]'], 'user': ['[class*="user"]', '[class*="profile"]'], 'name': ['[class*="name"]'],
            'date': ['[class*="date"]', '[class*="time"]', 'time'], 'timestamp': ['time'],
            'product': ['[class*="product"]', '[itemtype*="Product"]'], 'price': ['[class*="price"]'], 'rating': ['[class*="rating"]'],
            'image': ['img'], 'picture': ['picture'], 'photo': ['img', 'figure'], 'gallery': ['[class*="gallery"]'],
            'link': ['a'], 'url': ['a[href]'], 'href': ['a[href]'], 'text': ['p', 'div > p'], 'content': ['p']
        }
        
        active_selectors = set()
        prompt_words = set(prompt.split())

        for keyword, selectors in structural_keywords.items():
            if keyword in prompt_words:
                active_selectors.update(selectors)
        
        relevant_elements = []
        if active_selectors:
            for selector in active_selectors:
                elements = search_scope.select(selector, limit=20)
                if elements:
                    logger.info(f"Found {len(elements)} priority elements for selector '{selector}' based on keyword '{keyword}'.")
                    relevant_elements.extend(elements)

        # --- Stage 3: Build the final HTML snippet to send. ---
        output_parts = []
        head = soup.find('head')
        if head:
            title = head.find('title')
            if title and title.string:
                output_parts.append(str(title))
            meta_desc = head.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                output_parts.append(str(meta_desc))

        # If I found a main content area but no specific elements, I'll just use that.
        if main_content_soup and not relevant_elements:
            logger.info("No specific elements found, using the main content area as context.")
            output_parts.append(str(main_content_soup))
        else:
            # I'll grab the parents of the elements I found to give the AI more context.
            contextual_elements = []
            unique_elements = list(dict.fromkeys(relevant_elements)) # No duplicates, keep order.
            
            for element in unique_elements:
                # Go up two levels to the grandparent for even more context.
                parent = element.parent
                grandparent = parent.parent if parent else None
                if grandparent and grandparent.name not in ['body', 'html']:
                    contextual_elements.append(grandparent)
                elif parent and parent.name not in ['body', 'html']: # Or just the parent.
                    contextual_elements.append(parent)
            
            # Get rid of duplicate parents.
            unique_contextual_elements = list(dict.fromkeys(contextual_elements))
            logger.info(f"Extracted {len(unique_contextual_elements)} unique contextual parent/grandparent elements for the LLM.")
            for element in unique_contextual_elements[:30]: # Limit how many blocks I send.
                output_parts.append(str(element))

        # If I still have almost nothing, just grab the beginning of the body.
        if len(output_parts) <= 2:
             body = soup.find('body')
             if body:
                 logger.info("Fallback: No relevant content found, using start of body.")
                 output_parts.append(str(body)[:15000])

        combined_content = '\n\n'.join(output_parts)
        
        # Don't send a huge amount of HTML.
        max_len = 75000
        if len(combined_content) > max_len:
            combined_content = combined_content[:max_len] + "\n\n... (HTML truncated)"

        logger.info(f"Extracted {len(combined_content)} characters of relevant HTML content based on prompt analysis.")
        return combined_content
        
    except Exception as e:
        logger.warning(f"Error during intelligent HTML extraction: {e}, using raw content fallback.")
        return html_content[:15000]


def validate_scraper_results(stdout: str, user_prompt: str) -> Dict[str, any]:
    # After the scraper runs, I need to check if the output is any good.
    
    try:
        if not stdout.strip():
            return {
                "valid": False,
                "feedback": "The scraper produced no output. This usually means no data was found or there was an error."
            }
        
        # See if it's valid JSON.
        try:
            data = json.loads(stdout)
            if isinstance(data, list):
                item_count = len(data)
                logger.info(f"Scraper returned {item_count} items")

                if item_count == 0:
                    return {
                        "valid": False,
                        "feedback": "The scraper returned an empty list. This could mean the CSS selectors are wrong or the data is loaded dynamically. Please try again."
                    }
                
                # Check if the user asked for a certain number of items.
                import re
                number_match = re.search(r'(\d+)', user_prompt)
                if number_match:
                    requested_count = int(number_match.group(1))
                    if item_count == 0:
                        return {
                            "valid": False,
                            "feedback": f"No items were found. Expected {requested_count} items. The website structure might have changed or the selectors need adjustment."
                        }
                    elif item_count < requested_count * 0.5:  # Less than 50% of requested
                        return {
                            "valid": False,
                            "feedback": f"Only found {item_count} items, but {requested_count} were requested. The scraper may need better selectors."
                        }
                
                # If the user wanted images, check for URLs.
                if any(keyword in user_prompt.lower() for keyword in ['image', 'picture', 'photo', 'img']):
                    if item_count > 0:
                        sample_item = data[0] if isinstance(data[0], dict) else {"url": data[0]}
                        has_urls = any('url' in str(item).lower() or 'http' in str(item) for item in data[:3])
                        if not has_urls:
                            return {
                                "valid": False,
                                "feedback": "Found data but no URLs detected. For image scraping, expected to find image URLs."
                            }
                
                return {
                    "valid": True,
                    "feedback": f"Successfully extracted {item_count} items"
                }
            
            elif isinstance(data, dict):
                if not data:
                    return {
                        "valid": False,
                        "feedback": "The scraper returned an empty JSON object. The selectors might be incorrect."
                    }
                return {
                    "valid": True,
                    "feedback": "Successfully extracted structured data"
                }
            
        except json.JSONDecodeError:
            # If it's not JSON, maybe it's just plain text. That's okay sometimes.
            lines = stdout.strip().split('\n')
            if len(lines) > 0 and any(line.strip() for line in lines):
                return {
                    "valid": True,
                    "feedback": f"Successfully extracted {len(lines)} lines of data"
                }
        
        return {
            "valid": False,
            "feedback": "The scraper output doesn't appear to contain structured data"
        }
        
    except Exception as e:
        logger.error(f"Error validating scraper results: {e}")
        return {
            "valid": False,
            "feedback": f"Error validating results: {str(e)}"
        }


def cleanup_old_temp_files(max_age_hours: int = 24):
    # I should clean up old temp files so they don't fill up the disk.
    
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        working_dir = os.getcwd()
        
        # Clean up the temp scraper folders.
        temp_base_dir = os.path.join(working_dir, "temp_scrapers")
        if os.path.exists(temp_base_dir):
            for item in os.listdir(temp_base_dir):
                item_path = os.path.join(temp_base_dir, item)
                if os.path.isdir(item_path):
                    item_age = current_time - os.path.getctime(item_path)
                    if item_age > max_age_seconds:
                        shutil.rmtree(item_path)
                        logger.info(f"Cleaned up old temp directory: {item}")
        
        # Clean up the old zip files.
        downloads_dir = os.path.join(working_dir, "downloads")
        if os.path.exists(downloads_dir):
            for item in os.listdir(downloads_dir):
                if item.endswith('.zip'):
                    item_path = os.path.join(downloads_dir, item)
                    item_age = current_time - os.path.getctime(item_path)
                    if item_age > max_age_seconds:
                        os.remove(item_path)
                        logger.info(f"Cleaned up old download file: {item}")
                        
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def run_scraping_job(user_prompt: str, url: str, settings: Dict[str, Any], log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, any]:
    # This is the main function that does everything.
    # It gets the HTML, generates the code, tests it, and repeats if it fails.
    
    # Give each job a unique ID.
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    def log(message):
        logger.info(message)
        if log_callback:
            log_callback(message)

    log(f"Starting scraping job with run_id: {run_id}")
    log(f"Target URL: {url}")
    
    # I'll create a temp folder for this job.
    working_dir = os.getcwd()
    temp_base_dir = os.path.join(working_dir, "temp_scrapers")
    os.makedirs(temp_base_dir, exist_ok=True)
    
    temp_dir = os.path.join(temp_base_dir, run_id)
    
    logger.info(f"Starting scraping job with run_id: {run_id}")
    logger.info(f"Target URL: {url}")
    logger.info(f"User prompt: {user_prompt}")
    logger.debug(f"Temporary directory: {temp_dir}")
    
    # Clean up old files before I start.
    cleanup_old_temp_files(24)
    
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Get the HTML from the URL.
        log(f"Fetching content from: {url}")
        with httpx.Client(
            timeout=30.0, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            html_content = response.text
            log(f"Successfully fetched {len(html_content)} characters from URL")
        
        # Create the structure map for the first try.
        structure_map = create_html_structure_map(html_content, log_callback)

        # Get the most relevant HTML for the first try.
        relevant_html = extract_relevant_html(html_content, user_prompt)
        log(f"Extracted {len(relevant_html)} characters of relevant content for analysis")
        
        # I'll try a few times before giving up.
        max_attempts = 5
        history = []
        
        for attempt in range(max_attempts):
            log(f"Attempt {attempt + 1}/{max_attempts}: Generating scraping script...")
            
            # If this isn't the first try, get more HTML context.
            if attempt > 0 and history:
                log("Expanding HTML context for retry...")
                relevant_html = _expand_html_context(html_content, history[-1]['html_snippet'], log_callback)
                log(f"Expanded HTML context to {len(relevant_html)} characters.")
                # Don't need the map on retries.
                structure_map_for_request = None
            else:
                structure_map_for_request = structure_map

            try:
                # Call the AI to generate the script.
                result = generate_script(
                    prompt=user_prompt, 
                    url=url, 
                    html_content=relevant_html, 
                    settings=settings, 
                    history=history, 
                    structure_map=structure_map_for_request,
                    log_callback=log_callback
                )
                current_code = result["scraper_py"]
                
                log("Script generation completed, writing files...")
                
                # Write the generated files to the temp folder.
                scraper_path = os.path.join(temp_dir, "scraper.py")
                requirements_path = os.path.join(temp_dir, "requirements.txt")
                
                logger.debug(f"Writing scraper.py to: {scraper_path}")
                with open(scraper_path, "w", encoding="utf-8") as f:
                    f.write(result["scraper_py"])
                
                logger.debug(f"Writing requirements.txt to: {requirements_path}")
                with open(requirements_path, "w", encoding="utf-8") as f:
                    f.write(result["requirements_txt"])
                
                log("Files written, preparing to execute test...")
                
                # Run the script I just created.
                execution_result = run_script_in_dir(temp_dir, log_callback)
                
                # Check how it went.
                exit_code = execution_result["exit_code"]
                stdout = execution_result["stdout"]
                stderr = execution_result["stderr"]
                
                log(f"Script execution finished with exit code: {exit_code}")
                
                if exit_code == 0 and stdout.strip():
                    # It ran, now check if the output is good.
                    log("Validating script output...")
                    validation = validate_scraper_results(stdout, user_prompt)
                    log(f"Validation result: {validation['feedback']}")
                    
                    if validation["valid"]:
                        # Success!
                        log("Success! Script executed and output is valid.")
                        
                        # Zip it up for the user to download.
                        zip_path = _create_zip_package(temp_dir, run_id)
                        log(f"Created downloadable package: {run_id}.zip")
                        
                        return {
                            "status": "success",
                            "message": "Scraper generated and tested successfully!",
                            "data_preview": _truncate_preview(stdout),
                            "download_url": f"/api/download/{run_id}",
                            "zip_path": zip_path
                        }
                    else:
                        # The output wasn't good, so I'll try again.
                        failure_reason = validation['feedback']
                        history.append({
                            "reason": failure_reason,
                            "code": current_code,
                            "stdout": _truncate_preview(stdout, 2000),
                            "stderr": _truncate_preview(stderr, 2000),
                            "html_snippet": relevant_html
                        })
                        log(f"Results validation failed. Retrying...")
                        
                        if attempt == max_attempts - 1:
                            # I've failed too many times.
                            logger.error("All attempts failed validation")
                            break
                        else:
                            continue
                elif "playwright install" in stderr.lower():
                    # This means the user needs to do something.
                    log("Script requires Playwright installation by the user.")
                    
                    # Zip it up so they can download it.
                    zip_path = _create_zip_package(temp_dir, run_id)
                    log(f"Created ZIP package for Playwright script: {zip_path}")
                    
                    return {
                        "status": "action_required",
                        "message": "The script requires Playwright. Please run 'playwright install' and then run the script from the downloaded zip.",
                        "download_url": f"/api/download/{run_id}",
                        "zip_path": zip_path
                    }
                
                else:
                    # The script failed to run, so I'll try again.
                    failure_reason = f"Script execution failed with exit code {exit_code}. See STDERR for details."
                    history.append({
                        "reason": failure_reason,
                        "code": current_code,
                        "stdout": _truncate_preview(stdout, 2000),
                        "stderr": _truncate_preview(stderr, 2000),
                        "html_snippet": relevant_html
                    })
                    
                    log(f"Script failed on attempt {attempt + 1}. Retrying with error context.")
                    
                    if attempt == max_attempts - 1:
                        # I've failed too many times.
                        log("All attempts failed.")
                        break
            
            except Exception as e:
                log(f"An error occurred during attempt {attempt + 1}: {str(e)}")
                # Something unexpected happened, add it to the history.
                history.append({
                    "reason": f"An unexpected error occurred during script generation or execution: {str(e)}",
                    "code": current_code if 'current_code' in locals() else "Code not generated",
                    "stdout": "",
                    "stderr": str(e),
                    "html_snippet": relevant_html
                })
                
                if attempt == max_attempts - 1:
                    break
        
        # I've tried everything and it still failed.
        log("Scraping job failed after all attempts.")
        final_error_details = history[-1] if history else {"reason": "Unknown failure", "stderr": "No details available."}
        return {
            "status": "error",
            "message": "The agent failed to generate a working scraper. Please check the website or try a more specific prompt.",
            "final_error": f"Final attempt failed: {final_error_details['reason']}\n\nSTDERR:\n{final_error_details['stderr']}"
        }
    
    except httpx.RequestError as e:
        log(f"HTTP request error: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to fetch content from URL: {str(e)}"
        }
    except Exception as e:
        log(f"Unexpected error in scraping job: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error occurred: {str(e)}"
        }
    finally:
        # I'm not cleaning up the temp folder here, because the user might need to download the file.
        # Cleanup will happen later.
        log(f"Scraping job processing finished for run_id: {run_id}")


def _create_zip_package(temp_dir: str, run_id: str) -> str:
    # This zips up all the files for the user to download.
    
    # Make sure the downloads folder exists.
    working_dir = os.getcwd()
    downloads_dir = os.path.join(working_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    zip_path = os.path.join(downloads_dir, f"{run_id}.zip")
    logger.debug(f"Creating ZIP package at: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            file_count = 0
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
                    file_count += 1
                    logger.debug(f"Added to ZIP: {arcname}")
            
            logger.info(f"ZIP package created with {file_count} files")
        
        return zip_path
    except Exception as e:
        logger.error(f"Failed to create ZIP package: {str(e)}", exc_info=True)
        raise


def _truncate_preview(data: str, max_length: int = 500) -> str:
    # I don't want to show a huge wall of text in the preview.
    if len(data) <= max_length:
        return data
    return data[:max_length] + "..."