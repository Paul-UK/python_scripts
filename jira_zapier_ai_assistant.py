import json
import requests

""" 
#local testing

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class GeminiClient:
    def __init__(self, api_key=None, jira_token=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.jira_token = jira_token or os.getenv("JIRA_TOKEN")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")
        self.base_api_url = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/")
        self.REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
"""

class GeminiClient:
    def __init__(self, api_key, jira_token=None):
        self.api_key = api_key
        self.jira_token = jira_token
        self.model = "gemini-2.5-flash-preview-05-20"  # Single model for all tasks
        self.base_api_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.REQUEST_TIMEOUT = 60 
        self.temperature = 0.1  # lower temperature for most deterministic responses
        self.max_tokens = 2000  # Limit response length
        # top_p is irrelevant if top_k is 1, but set to 1.0 to be safe
        self.top_p = 1
        self.top_k = 1.0
        
        # Grounding config with google search
        self.standard_grounding_tool = {
                "google_search": {}
        }
        
        # Add product mapping this is subject to change but the cost of running Jira and parsing is not worth it
        self.product_mapping = {
            'Agreement calculator': '10729',
            'Alignerr connect': '11556',
            'API keys': '10553',
            'Attachment panel': '10727',
            'Audio editor': '10739',
            'Batches': '10544',
            'Benchmarks': '10505',
            'Billing and annotation counts dashboard': '10751',
            'Boost in-app services': '10746',
            'Bulk classification': '10839',
            'Catalog': '10514',
            'Classifications': '10741',
            'Cloud bucket': '11041',
            'Community': '11042',
            'Consensus': '10504',
            'Conversational editor': '10733',
            'CORS': '10567',
            'Data row details (editor)': '10743',
            'Data row import & file uploads': '10541',
            'Data row processing': '10756',
            'Data Rows tab': '10509',
            'Dataset management (UI)': '10929',
            'Delegated access': '10605',
            'Deletion request': '10683',
            'Deployments': '10753',
            'Document editor': '10734',
            'Editor adjustments': '10744',
            'Embeddings': '10561',
            'Entitlements': '10752',
            'Entity tool': '10735',
            'Exports': '10519',
            'Foundry': '10871',
            'GraphQL': '10551',
            'Ground truth import': '10754',
            'Hotkeys': '10745',
            'HTML editor': '10873',
            'Image editor': '10736',
            'Image overlays': '10742',
            'Issues and comments': '10508',
            'Labeling queue': '10507',
            'LBU': '10789',
            'LBU limits enforcement': '10901',
            'LLM editors': '10970',
            'LPOs': '10747',
            'MAL': '10512',
            'Metadata': '10515',
            'Metrics service': '10731',
            'Model': '10516',
            'Model error analysis': '10730',
            'Monitor': '11555',
            'Multi modal chat editor': '11165',
            'Notifications': '10903',
            'Ontology management': '10518',
            'Organization management (workspaces)': '10710',
            'Payments (Stripe integration)': '10902',
            'Performance dashboard': '10510',
            'Project creation flow': '10748',
            'Project overview': '10540',
            'Projects list page': '10749',
            'SDK - Python': '10564',
            'Sign in / sign out / sign up': '10542',
            'Similarity': '10562',
            'Solutions integrations/connectors': '10828',
            'SSO': '10517',
            'Tags': '10554',
            'Text editor': '10732',
            'Thumbnails': '10563',
            'Tiled editor': '10738',
            'User management': '10606',
            'Video editor': '10737',
            'Webhooks': '10555',
            'Workflow queues': '10506',
            'Workflow tasks': '10750'
        }
        
        # Add cause mapping
        self.cause_mapping = {
            'Alignerr': '11240',
            'Bug': '10495',
            'Documentation': '10496',
            'Usability': '10497',
            'Feature Request': '10498',
            'Transient Issue': '10499',
            'Request of MLSE': '10538',
            'None applicable': '10728'
        }
        
    def call_gemini_api(self, payload, response_schema=None):
        """Makes a POST request to the Gemini API with timeout and error handling"""
        url = f"{self.base_api_url}{self.model}:generateContent?key={self.api_key}"
        generation_config = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
            "topP": self.top_p,
            "topK": self.top_k,
        }
        if response_schema:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = response_schema
            # DO NOT add tools if using structured output
            payload["generationConfig"] = generation_config
        else:
            payload["generationConfig"] = generation_config
            payload["tools"] = [self.standard_grounding_tool]
        headers = {'Content-Type': 'application/json'}
        try:
            print(f"DEBUG: Sending API request to {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT)
            print(f"DEBUG: API response status code: {response.status_code}")
            response.raise_for_status()
            
            json_response = response.json()
            print(f"DEBUG: Response JSON structure keys: {list(json_response.keys())}")
            return json_response
        except requests.exceptions.Timeout:
            print("ERROR: API Request Timed Out")
            print(f"ERROR: Request timeout was set to {self.REQUEST_TIMEOUT} seconds")
            return {"error": f"API request timed out after {self.REQUEST_TIMEOUT} seconds", "raw_output": "Timeout occurred"}
        except requests.exceptions.RequestException as e:
            import traceback
            print(f"ERROR: API Request Failed: {e}")
            print(f"ERROR: Request traceback: {traceback.format_exc()}")
            error_text = "No response body"
            raw_output = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_text = json.dumps(error_json, indent=2)
                    print(f"ERROR: API Error Response: {error_text}")
                    raw_output = error_text
                    return {"error": f"API error: {error_json.get('error', {}).get('message', str(e))}", "raw_output": raw_output}
                except:
                    error_text = e.response.text
                    print(f"ERROR: API Error Body (non-JSON): {error_text}")
                    raw_output = error_text
            return {"error": f"API request failed: {str(e)}", "raw_output": raw_output}
        except Exception as e:
            import traceback
            print(f"ERROR: Unexpected error in API call: {e}")
            print(f"ERROR: Unexpected error traceback: {traceback.format_exc()}")
            return {"error": f"Unexpected error in API call: {str(e)}", "raw_output": str(e)}

    def get_image_content(self, image_url):
        """Get image content from URL"""
        try:
            print(f"DEBUG: Attempting to fetch from URL: {image_url}")
            
            headers = {}
            if self.jira_token:
                headers['Authorization'] = f'{self.jira_token}'
                print("DEBUG: Using authorization token for image request")
            
            response = requests.get(image_url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            print(f"DEBUG: Image fetch response status code: {response.status_code}")
            response.raise_for_status()
            
            # Optional: Log content type for debugging but don't validate
            content_type = response.headers.get('content-type', '')
            print(f"DEBUG: Content-Type: {content_type}, Content size: {len(response.content)} bytes")
            
            return response.content
            
        except Exception as e:
            print(f"ERROR: Failed to fetch image: {e}")
            return None

    def get_product_id(self, product_name):
        """Get the ID for a given product name using exact matching"""
        # Clean up the product name (remove extra spaces, match case)
        cleaned_name = product_name.strip()
        
        # Try exact match first
        if cleaned_name in self.product_mapping:
            return self.product_mapping[cleaned_name]
        
        # Try case-insensitive match if exact match fails
        for key in self.product_mapping:
            if key.lower() == cleaned_name.lower():
                return self.product_mapping[key]
        
        return None

    def get_cause_id(self, cause_name):
        """Get the ID for a given cause using exact matching"""
        # Clean up the cause name (remove extra spaces, match case)
        cleaned_name = cause_name.strip()
        
        # Try exact match first
        if cleaned_name in self.cause_mapping:
            return self.cause_mapping[cleaned_name]
        
        # Try case-insensitive match if exact match fails
        for key in self.cause_mapping:
            if key.lower() == cleaned_name.lower():
                return self.cause_mapping[key]
        
        return None

    def analyze_content(self, prompt, image_data=None):
        """
        Analyze content with text and optional image input.
        Returns a dict with product (ID), cause (ID), and analysis.
        """
        try:
            parts = [{"text": prompt}]
            if image_data:
                total_size = len(prompt.encode('utf-8')) + len(image_data['content'].encode('utf-8'))
                # 7MB Gemini API limit for images + 3MB for prompt text
                if total_size > 10 * 1024 * 1024:
                    print("Combined prompt and image size exceeds limit")
                    return {"error": "Combined prompt and image size exceeds limit"}
                parts.append({
                    "inline_data": {
                        "mime_type": image_data['mime_type'],
                        "data": image_data['content']
                    }
                })

            payload = {"contents": [{"parts": parts}]}

            # Schema expects only the values, not the mapping keys
            response_schema = {
                "type": "object",
                "properties": {
                    "LS: Specific Product": {"type": "string"},
                    "LS: Cause for ticket": {"type": "string"},
                    "Labelbox Internal Analysis": {"type": "string"}
                },
                "required": [
                    "LS: Specific Product",
                    "LS: Cause for ticket",
                    "Labelbox Internal Analysis"
                ],
                "propertyOrdering": [
                    "LS: Specific Product",
                    "LS: Cause for ticket",
                    "Labelbox Internal Analysis"
                ]
            }

            print(f"Sending payload: {json.dumps(payload, indent=2)}")
            response_data = self.call_gemini_api(payload, response_schema=response_schema)
            print(f"Complete raw model response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")

            if isinstance(response_data, dict) and 'error' in response_data:
                return response_data

            # Capture raw text for error reporting
            raw_model_output = ""
            if response_data and 'candidates' in response_data and response_data['candidates']:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                    part = candidate['content']['parts'][0]
                    if 'text' in part:
                        raw_model_output = part['text']

            # Enhanced debugging for response parsing
            parsed_response = None
            if not response_data:
                print("ERROR: response_data is None or empty")
                return {"error": "Empty response from API"}
                
            if 'candidates' not in response_data:
                print(f"ERROR: No 'candidates' field in response. Keys found: {list(response_data.keys())}")
                return {"error": "No candidates field in API response", "raw_output": json.dumps(response_data)}
                
            if not response_data['candidates']:
                print("ERROR: 'candidates' array is empty")
                return {"error": "Empty candidates array in API response", "raw_output": json.dumps(response_data)}
            
            # Get the first candidate
            candidate = response_data['candidates'][0]
            print(f"DEBUG: First candidate: {json.dumps(candidate, indent=2)}")
            
            if 'content' not in candidate:
                print(f"ERROR: No 'content' field in candidate. Keys found: {list(candidate.keys())}")
                return {"error": "No content field in candidate", "raw_output": json.dumps(candidate)}
                
            content = candidate['content']
            print(f"DEBUG: Content: {json.dumps(content, indent=2)}")
            
            if 'parts' not in content or not content['parts']:
                print(f"ERROR: Missing or empty 'parts' in content. Content keys: {list(content.keys())}")
                return {"error": "Missing parts in content", "raw_output": json.dumps(content)}
                
            part = content['parts'][0]
            print(f"DEBUG: First part: {json.dumps(part, indent=2)}")
            
            # Try multiple approaches to extract the structured data
            if 'text' in part:
                # If text is already a valid JSON string
                text = part['text']
                print(f"DEBUG: Text content found: {text[:500]}...")  # Print first 500 chars
                
                try:
                    # In case it's a JSON string
                    parsed_response = json.loads(text)
                    print("DEBUG: Successfully parsed text as JSON")
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to parse text as JSON: {e}")
                    # Try to find JSON-like content within the text
                    try:
                        # Look for content between { and }
                        import re
                        json_match = re.search(r'\{.*\}', text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            print(f"DEBUG: Extracted JSON-like string: {json_str[:500]}...")
                            parsed_response = json.loads(json_str)
                            print("DEBUG: Successfully parsed extracted JSON")
                        else:
                            print("ERROR: No JSON-like pattern found in text")
                    except Exception as e2:
                        print(f"ERROR: Failed to extract JSON from text: {e2}")
                    
                # If text parsing failed but text is a dict
                if not parsed_response and isinstance(text, dict):
                    parsed_response = text
                    print("DEBUG: Text was already a dict")
            else:
                print(f"ERROR: No 'text' field in part. Part keys: {list(part.keys())}")
                # In some cases, the JSON might be directly in the part
                try:
                    if isinstance(part, dict) and "LS: Specific Product" in part:
                        parsed_response = part
                        print("DEBUG: Used part object directly as parsed_response")
                except Exception as e:
                    print(f"ERROR: Failed to use part directly: {e}")

            if not parsed_response and 'functionResponse' in part:
                print("DEBUG: Trying to extract from functionResponse")
                try:
                    func_response = part['functionResponse']
                    parsed_response = json.loads(func_response)
                    print("DEBUG: Successfully extracted from functionResponse")
                except Exception as e:
                    print(f"ERROR: Failed to parse functionResponse: {e}")

            if not parsed_response:
                print("ERROR: Failed to extract structured data from response after all attempts")
                # Print the raw text response to help diagnose the issue
                if response_data and 'candidates' in response_data and response_data['candidates']:
                    candidate = response_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                        part = candidate['content']['parts'][0]
                        if 'text' in part:
                            raw_text = part['text']
                            print(f"ERROR: Raw model output text:\n{'-'*50}\n{raw_text}\n{'-'*50}")
                        else:
                            print(f"ERROR: No 'text' in part. Raw part content:\n{json.dumps(part, indent=2)}")
                print(f"ERROR: Full raw response structure:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
                return {"error": "Failed to extract structured data from response", "raw_output": raw_model_output or json.dumps(response_data)}

            # Map product/cause names to IDs
            product_name = parsed_response.get("LS: Specific Product", "").strip()
            cause_name = parsed_response.get("LS: Cause for ticket", "").strip()
            analysis = parsed_response.get("Labelbox Internal Analysis", "")

            print(f"Extracted values - Product: {product_name}, Cause: {cause_name}")

            product_id = self.get_product_id(product_name) or ""
            cause_id = self.get_cause_id(cause_name) or ""

            return {
                "product": product_id,
                "cause": cause_id,
                "analysis": analysis
            }

        except Exception as e:
            import traceback
            print(f"Error in content analysis: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Error in content analysis: {str(e)}"}

    def analyze_issue(self, issue_key, summary, description, attachment_url=None):
        """Analyzes issue content and attachment if provided"""
        try:
            print(f"DEBUG: Processing issue: {issue_key}")
            
            # Construct the prompt
            prompt_text = (
                "Analyze the following Jira issue and return a structured response in the exact JSON format specified below.\n\n"
                "1. **Analysis Required:**\n"
                "* Classify LS: Specific Product, pick one of the options: [Agreement calculator, Alignerr connect, API keys, Attachment panel, Audio editor, Batches, Benchmarks, Billing and annotation counts dashboard, Boost in-app services, Bulk classification, Catalog, Classifications, Cloud bucket, Community, Consensus, Conversational editor, CORS, Data row details (editor), Data row import & file uploads, Data row processing, Data row tab, Dataset management (UI), Delegated access, Deletion request, Deployments, Document editor, Editor adjustments, Embeddings, Entitlements, Entity tool, Exports, Foundry, GraphQL, Ground truth import, Hotkeys, HTML editor, Image editor, Image overlays, Issues and comments, Labeling queue, LBU, LBU limits enforcement, LLM Editors, MAL, Metadata, Metrics service, Model, Model error analysis, Monitor, Multi modal chat editor, Notifications, Ontology management, Organization management (workspaces), Payments (Stripe integration), Performance dashboard, Project creation flow, Project overview, Projects list page, SDK - Python, Sign in / sign out / sign up, Similarity, Solutions integrations/connectors, SSO, Tags, Text editor, Thumbnails, Tiled editor, User management, Video editor, Webhooks, Workflow queues, Workflow tasks]\n"
                "* Classify LS: Cause for ticket, pick one of the options: [Bug, Documentation, Usability, Feature Request, Transient Issue, Request of MLSE, None applicable, Alignerr]\n\n"
                f"Issue: {issue_key}\nSummary: {summary}\nDescription: {description}\n\n"
                "Provide a technical analysis that includes in less than 100 words:\n"
                "1. A clear root cause analysis focusing on Labelbox components (UI/SDK/API)\n"
                "2. Specific areas within Labelbox to investigate (list 2-3 key areas)\n"
                "3. Concrete next steps for investigation (list 2-3 steps)\n\n"
                "Return the result as a single JSON object using the following structure:\n"
                "```json\n"
                "{\n"
                '  "LS: Specific Product": "<exact match from the product list>",\n'
                '  "LS: Cause for ticket": "<exact match from the cause list>",\n'
                '  "Labelbox Internal Analysis": "* Root Cause: Based on the issue description, the problem appears to be [specific technical explanation]\\n* Key Areas: 1) [specific component/area], 2) [specific component/area], 3) [specific component/area]\\n* Next Steps: 1) [specific action], 2) [specific action], 3) [specific action]"\n'
                "}\n"
                "```\n"
                "Ensure the response is a valid JSON object exactly matching this structure. Use only the exact values from the provided lists for product and cause classifications. Replace the bracketed text with actual analysis based on the issue details."
            )
            
            print(f"DEBUG: Prompt length: {len(prompt_text)} characters")
            
            # Process attachment if present
            if attachment_url:
                print(f"DEBUG: Processing attachment URL: {attachment_url}")
                image_data = self.get_image_content(attachment_url)
                if image_data:
                    print("DEBUG: Successfully processed attachment, including in analysis")
                    result = self.analyze_content(prompt_text, image_data)
                else:
                    print("ERROR: Failed to process attachment content, proceeding with text-only analysis")
                    result = self.analyze_content(prompt_text)
            else:
                print("DEBUG: No attachment provided, proceeding with text-only analysis")
                result = self.analyze_content(prompt_text)
            
            print(f"DEBUG: Analysis result: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Error analyzing issue: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"ERROR: Issue analysis traceback: {traceback.format_exc()}")
            return {"error": error_msg, "raw_output": f"Exception during analysis: {str(e)}\n\nTraceback: {traceback.format_exc()}"}

# Main execution block
try:
    # Get individual values from input_data
    issue_key = input_data.get('issue_key')
    summary = input_data.get('summary', '')
    description = input_data.get('description', '')
    jira_token =''
    api_key_gemini = ''
    
    # Get attachment URL if available
    attachments = input_data.get('attachments', [])
    attachment_url = attachments if attachments else None
    
    # Debug logging
    print("DEBUG: Input data received:")
    print(f"DEBUG: Issue Key: {issue_key}")
    print(f"DEBUG: Summary: {summary}")
    print(f"DEBUG: Description length: {len(description) if description else 0} chars")
    print(f"DEBUG: Attachment URL: {attachment_url}")
    
    if not all([issue_key, summary, description]):
        error_msg = "Missing required issue information"
        print(f"ERROR: {error_msg}")
        output = {"error": error_msg}
    else:
        print("DEBUG: Creating GeminiClient instance")
        gemini_client = GeminiClient(api_key_gemini, jira_token)
        print("DEBUG: Calling analyze_issue method")
        analysis = gemini_client.analyze_issue(issue_key, summary, description, attachment_url)
        if analysis:
            if "error" in analysis:
                print(f"ERROR: Analysis returned an error: {analysis['error']}")
                output = {"error": analysis["error"], "raw_output": analysis.get("raw_output", "No raw output captured")}
            else:
                print("DEBUG: Analysis completed successfully")
                output = {"analysis": analysis}
        else:
            error_msg = "Analysis returned no results"
            print(f"ERROR: {error_msg}")
            output = {"error": error_msg}
            
    print(f"DEBUG: Final output: {json.dumps(output, indent=2)}")
    
except Exception as e:
    import traceback
    error_msg = str(e)
    print(f"ERROR: Error in main execution: {error_msg}")
    print(f"ERROR: Main execution traceback: {traceback.format_exc()}")
    output = {"error": error_msg}
