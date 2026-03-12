import json
import requests
from typing import Dict, List, Optional
import traceback
import os
from datetime import datetime

# Add timestamp to debug logs
def debug_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"DEBUG [{timestamp}]: {message}")

def error_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"ERROR [{timestamp}]: {message}")
    
class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gemini-2.0-flash-lite"  # Single model for all tasks
        self.base_api_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.REQUEST_TIMEOUT = 60  # seconds
        self.temperature = 0.1  # lower temperature for most deterministic responses
        self.max_tokens = 2000  # Limit response length
        self.top_p = 1
        self.top_k = 1.0
        debug_log(f"GeminiClient initialized with model: {self.model}, timeout: {self.REQUEST_TIMEOUT}s")
        
    def call_gemini_api(self, payload: Dict, response_schema: Optional[Dict] = None) -> Dict:
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
        headers = {'Content-Type': 'application/json'}
        
        debug_log(f"Preparing Gemini API request to {url}")
        debug_log(f"Payload size: {len(json.dumps(payload))} characters")
        
        try:
            debug_log(f"Sending API request to {url}")
            start_time = datetime.now()
            response = requests.post(url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            debug_log(f"API request completed in {duration:.2f} seconds with status code: {response.status_code}")
            response.raise_for_status()
            
            json_response = response.json()
            debug_log(f"Response JSON structure keys: {list(json_response.keys())}")
            debug_log(f"Gemini response size: {len(json.dumps(json_response))} characters")
            return json_response
        except requests.exceptions.Timeout:
            error_log(f"API Request Timed Out after {self.REQUEST_TIMEOUT} seconds")
            return {"error": f"API request timed out after {self.REQUEST_TIMEOUT} seconds", "raw_output": "Timeout occurred"}
        except requests.exceptions.RequestException as e:
            error_log(f"API Request Failed: {e}")
            error_log(f"Request traceback: {traceback.format_exc()}")
            error_text = "No response body"
            raw_output = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_text = json.dumps(error_json, indent=2)
                    error_log(f"API Error Response: {error_text}")
                    raw_output = error_text
                    return {"error": f"API error: {error_json.get('error', {}).get('message', str(e))}", "raw_output": raw_output}
                except:
                    error_text = e.response.text
                    error_log(f"API Error Body (non-JSON): {error_text}")
                    raw_output = error_text
            return {"error": f"API request failed: {str(e)}", "raw_output": raw_output}
        except Exception as e:
            error_log(f"Unexpected error in API call: {e}")
            error_log(f"Unexpected error traceback: {traceback.format_exc()}")
            return {"error": f"Unexpected error in API call: {str(e)}", "raw_output": str(e)}


class JiraTicketAnalyzer:
    def __init__(self, jira_base_url: str, jira_email: str, jira_api_token: str, gemini_client: GeminiClient):
        self.jira_base_url = jira_base_url
        self.auth = (jira_email, jira_api_token)
        self.gemini_client = gemini_client
        debug_log(f"JiraTicketAnalyzer initialized with base URL: {jira_base_url}")

    def get_ticket_details(self, issue_key: str) -> Optional[Dict]:
        """Fetch full ticket details including comments and attachments"""
        try:
            url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}"
            params = {
                "expand": "comments,attachments"
            }
            debug_log(f"Fetching ticket details from {url} with expand params: {params}")
            start_time = datetime.now()
            response = requests.get(url, auth=self.auth, params=params)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            debug_log(f"Jira API request completed in {duration:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code != 200:
                error_log(f"Jira API returned non-200 status: {response.status_code}")
                error_log(f"Response body: {response.text[:1000]}...")  # Log first 1000 chars
            
            response.raise_for_status()
            data = response.json()
            debug_log(f"Ticket data retrieved successfully. Data size: {len(json.dumps(data))} characters")
            return data
        except Exception as e:
            error_log(f"Error fetching ticket details: {e}")
            error_log(f"{traceback.format_exc()}")
            return None

    def extract_ticket_content(self, ticket_data: Dict) -> Dict:
        """Extract relevant information from ticket data"""
        debug_log(f"Extracting content from ticket data")
        try:
            content = {
                "key": ticket_data["key"],
                "summary": ticket_data["fields"]["summary"],
                "description": ticket_data["fields"]["description"] or "",
                "status": ticket_data["fields"]["status"]["name"],
                "created": ticket_data["fields"]["created"],
                "updated": ticket_data["fields"]["updated"],
                "comments": [],
                "attachments": []
            }

            # Extract comments
            comment_count = 0
            if "comment" in ticket_data["fields"] and "comments" in ticket_data["fields"]["comment"]:
                comment_count = len(ticket_data["fields"]["comment"]["comments"])
                debug_log(f"Found {comment_count} comments in ticket")
                
                for comment in ticket_data["fields"]["comment"]["comments"]:
                    content["comments"].append({
                        "author": comment["author"]["displayName"],
                        "body": comment["body"],
                        "created": comment["created"]
                    })
            else:
                debug_log("No comments found in ticket data")

            # Extract attachments
            attachment_count = 0
            if "attachment" in ticket_data["fields"]:
                attachment_count = len(ticket_data["fields"]["attachment"])
                debug_log(f"Found {attachment_count} attachments in ticket")
                
                for attachment in ticket_data["fields"]["attachment"]:
                    content["attachments"].append({
                        "filename": attachment["filename"],
                        "created": attachment["created"],
                        "url": attachment["content"]
                    })
            else:
                debug_log("No attachments found in ticket data")
                
            debug_log(f"Extracted content for ticket {content['key']} - {comment_count} comments, {attachment_count} attachments")
            return content
        except KeyError as e:
            error_log(f"KeyError when extracting ticket content: {e}")
            error_log(f"Available fields: {list(ticket_data['fields'].keys())}")
            raise

    def generate_summary(self, ticket_content: Dict) -> str:
        """Generate summary using Gemini"""
        debug_log(f"Generating summary for ticket: {ticket_content['key']}")
        
        prompt_text = f"""Summarize this Jira ticket with a focus on clarity and accuracy. The summary should cover:
1. Key issue details
2. A chronological timeline of customer interactions
3. Relevant technical discussions or proposed solutions
4. Current resolution status

Ticket Information:
Key: {ticket_content['key']}
Summary: {ticket_content['summary']}
Description: {ticket_content['description'][:500]}... (truncated)
Status: {ticket_content['status']}

Comments History:
{self._format_comments(ticket_content['comments'])}

Attachments:
{self._format_attachments(ticket_content['attachments'])}
"""
        debug_log(f"Prompt length: {len(prompt_text)} characters")
        
        prompt = {
            "contents": [{
                "parts": [{
                    "text": prompt_text
                }]
            }]
        }

        debug_log(f"Calling Gemini API for ticket summary")
        response = self.gemini_client.call_gemini_api(prompt)
        
        if response and 'candidates' in response:
            summary = response['candidates'][0]['content']['parts'][0]['text']
            debug_log(f"Successfully generated summary ({len(summary)} chars)")
            debug_log(f"First 200 chars of summary: {summary[:200]}...")
            return summary
        else:
            error_log(f"Failed to generate summary. Response: {json.dumps(response)}")
            if 'error' in response:
                error_log(f"Error details: {response['error']}")
            return "Failed to generate summary"

    def _format_comments(self, comments: List[Dict]) -> str:
        formatted = "\n".join([
            f"- {comment['created']}: {comment['author']}: {comment['body'][:100]}..." if len(comment['body']) > 100 else f"- {comment['created']}: {comment['author']}: {comment['body']}"
            for comment in comments
        ])
        debug_log(f"Formatted {len(comments)} comments, total size: {len(formatted)} chars")
        return formatted

    def _format_attachments(self, attachments: List[Dict]) -> str:
        formatted = "\n".join([
            f"- {attachment['created']}: {attachment['filename']}"
            for attachment in attachments
        ])
        debug_log(f"Formatted {len(attachments)} attachments, total size: {len(formatted)} chars")
        return formatted

    def add_summary_comment(self, issue_key: str, summary: str) -> bool:
        """Add the generated summary as an internal comment"""
        try:
            url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}/comment"
            payload = {
                "body": f"*AI-Generated Ticket Summary*\n\n{summary}",
                "properties": [{"key": "sd.public.comment", "value": {"internal": True}}]
            }
            debug_log(f"Adding summary comment to {issue_key} with URL: {url}")
            debug_log(f"Comment payload size: {len(json.dumps(payload))} characters")
            
            start_time = datetime.now()
            response = requests.post(url, auth=self.auth, json=payload)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            debug_log(f"Jira comment API request completed in {duration:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code >= 200 and response.status_code < 300:
                debug_log(f"Successfully added comment to {issue_key}")
                debug_log(f"Comment response: {response.text[:500]}...")
                return True
            else:
                error_log(f"Failed to add comment. Status code: {response.status_code}")
                error_log(f"Response body: {response.text}")
                response.raise_for_status()
                return False
        except Exception as e:
            error_log(f"Error adding summary comment: {e}")
            error_log(f"{traceback.format_exc()}")
            return False


def handle_webhook(issue_key, jira_base_url, jira_email, jira_api_token, gemini_api_key):
    """Process a single Jira ticket and add an AI-generated summary comment"""
    debug_log(f"Processing issue_key: {issue_key}")
    
    try:
        # Initialize clients
        debug_log(f"Initializing GeminiClient")
        gemini_client = GeminiClient(api_key=gemini_api_key)
        
        debug_log(f"Initializing JiraTicketAnalyzer")
        analyzer = JiraTicketAnalyzer(
            jira_base_url=jira_base_url,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            gemini_client=gemini_client
        )

        # Get ticket details
        debug_log(f"Getting ticket details for {issue_key}")
        ticket_data = analyzer.get_ticket_details(issue_key)
        if not ticket_data:
            error_log(f"Failed to fetch ticket details for {issue_key}")
            return {
                "success": False, 
                "error": f"Failed to fetch ticket details for {issue_key}"
            }

        # Extract relevant content
        debug_log(f"Extracting content from ticket data")
        ticket_content = analyzer.extract_ticket_content(ticket_data)

        # Generate summary
        debug_log(f"Generating summary")
        summary = analyzer.generate_summary(ticket_content)

        # Add summary as internal comment
        debug_log(f"Adding summary comment")
        comment_added = analyzer.add_summary_comment(issue_key, summary)
        
        if comment_added:
            debug_log(f"Successfully added comment to ticket {issue_key}")
        else:
            error_log(f"Failed to add comment to ticket {issue_key}")

        debug_log(f"Processing complete for {issue_key}")
        return {
            "success": True,
            "issue_key": issue_key,
            "summary_added": comment_added,
            "summary": summary
        }
    except Exception as e:
        error_log(f"Unexpected error in handle_webhook: {e}")
        error_log(f"{traceback.format_exc()}")
        return {
            "success": False, 
            "error": str(e)
        }

# Main function that will be called by Zapier
def main(zapier):
    debug_log("Starting Zapier Code step execution")
    
    # Get input data from the trigger or previous steps
    trigger_data = zapier.trigger_output
    debug_log(f"Received trigger data: {json.dumps(trigger_data, default=str)[:1000]}...")
    
    # Extract the issue key from trigger data
    issue_key = issue_key = zapier.trigger_output["issue"]["key"]
    if not issue_key:
        error_log("Missing required issue_key in trigger data")
        return {"success": False, "error": "Missing required issue_key in trigger data"}
    
    # Get credentials from environment variables
    try:
        # Get credentials from environment variables
        jira_base_url = os.environ.get("JIRA_BASE_URL")
        jira_email = os.environ.get("JIRA_EMAIL") 
        jira_api_token = os.environ.get("JIRA_API_TOKEN")
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        # Validate credentials
        missing = []
        if not jira_base_url: missing.append("JIRA_BASE_URL")
        if not jira_email: missing.append("JIRA_EMAIL")
        if not jira_api_token: missing.append("JIRA_API_TOKEN")
        if not gemini_api_key: missing.append("GEMINI_API_KEY")
        
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            error_log(error_msg)
            return {"success": False, "error": error_msg}
            
        debug_log(f"Retrieved credentials from environment variables")
        
        # Process the Jira ticket
        result = handle_webhook(
            issue_key=issue_key,
            jira_base_url=jira_base_url,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            gemini_api_key=gemini_api_key
        )
        
        debug_log(f"Completed processing with result: {json.dumps(result, default=str)}")
        return result
        
    except Exception as e:
        error_log(f"Error in main Zapier function: {e}")
        error_log(traceback.format_exc())
        return {"success": False, "error": str(e)}

import json
import requests
from typing import Dict, List, Optional
import traceback
import os
from datetime import datetime

# Add timestamp to debug logs
def debug_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"DEBUG [{timestamp}]: {message}")

def error_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"ERROR [{timestamp}]: {message}")
    
class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gemini-2.0-flash-lite"  # Single model for all tasks
        self.base_api_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.REQUEST_TIMEOUT = 60  # seconds
        self.temperature = 0.1  # lower temperature for most deterministic responses
        self.max_tokens = 2000  # Limit response length
        self.top_p = 1
        self.top_k = 1.0
        debug_log(f"GeminiClient initialized with model: {self.model}, timeout: {self.REQUEST_TIMEOUT}s")
        
    def call_gemini_api(self, payload: Dict, response_schema: Optional[Dict] = None) -> Dict:
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
        headers = {'Content-Type': 'application/json'}
        
        debug_log(f"Preparing Gemini API request to {url}")
        debug_log(f"Payload size: {len(json.dumps(payload))} characters")
        
        try:
            debug_log(f"Sending API request to {url}")
            start_time = datetime.now()
            response = requests.post(url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            debug_log(f"API request completed in {duration:.2f} seconds with status code: {response.status_code}")
            response.raise_for_status()
            
            json_response = response.json()
            debug_log(f"Response JSON structure keys: {list(json_response.keys())}")
            debug_log(f"Gemini response size: {len(json.dumps(json_response))} characters")
            return json_response
        except requests.exceptions.Timeout:
            error_log(f"API Request Timed Out after {self.REQUEST_TIMEOUT} seconds")
            return {"error": f"API request timed out after {self.REQUEST_TIMEOUT} seconds", "raw_output": "Timeout occurred"}
        except requests.exceptions.RequestException as e:
            error_log(f"API Request Failed: {e}")
            error_log(f"Request traceback: {traceback.format_exc()}")
            error_text = "No response body"
            raw_output = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_text = json.dumps(error_json, indent=2)
                    error_log(f"API Error Response: {error_text}")
                    raw_output = error_text
                    return {"error": f"API error: {error_json.get('error', {}).get('message', str(e))}", "raw_output": raw_output}
                except:
                    error_text = e.response.text
                    error_log(f"API Error Body (non-JSON): {error_text}")
                    raw_output = error_text
            return {"error": f"API request failed: {str(e)}", "raw_output": raw_output}
        except Exception as e:
            error_log(f"Unexpected error in API call: {e}")
            error_log(f"Unexpected error traceback: {traceback.format_exc()}")
            return {"error": f"Unexpected error in API call: {str(e)}", "raw_output": str(e)}


class JiraTicketAnalyzer:
    def __init__(self, jira_base_url: str, jira_email: str, jira_api_token: str, gemini_client: GeminiClient):
        self.jira_base_url = jira_base_url
        self.auth = (jira_email, jira_api_token)
        self.gemini_client = gemini_client
        debug_log(f"JiraTicketAnalyzer initialized with base URL: {jira_base_url}")

    def get_ticket_details(self, issue_key: str) -> Optional[Dict]:
        """Fetch full ticket details including comments and attachments"""
        try:
            url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}"
            params = {
                "expand": "comments,attachments"
            }
            debug_log(f"Fetching ticket details from {url} with expand params: {params}")
            start_time = datetime.now()
            response = requests.get(url, auth=self.auth, params=params)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            debug_log(f"Jira API request completed in {duration:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code != 200:
                error_log(f"Jira API returned non-200 status: {response.status_code}")
                error_log(f"Response body: {response.text[:1000]}...")  # Log first 1000 chars
            
            response.raise_for_status()
            data = response.json()
            debug_log(f"Ticket data retrieved successfully. Data size: {len(json.dumps(data))} characters")
            return data
        except Exception as e:
            error_log(f"Error fetching ticket details: {e}")
            error_log(f"{traceback.format_exc()}")
            return None

    def extract_ticket_content(self, ticket_data: Dict) -> Dict:
        """Extract relevant information from ticket data"""
        debug_log(f"Extracting content from ticket data")
        try:
            content = {
                "key": ticket_data["key"],
                "summary": ticket_data["fields"]["summary"],
                "description": ticket_data["fields"]["description"] or "",
                "status": ticket_data["fields"]["status"]["name"],
                "created": ticket_data["fields"]["created"],
                "updated": ticket_data["fields"]["updated"],
                "comments": [],
                "attachments": []
            }

            # Extract comments
            comment_count = 0
            if "comment" in ticket_data["fields"] and "comments" in ticket_data["fields"]["comment"]:
                comment_count = len(ticket_data["fields"]["comment"]["comments"])
                debug_log(f"Found {comment_count} comments in ticket")
                
                for comment in ticket_data["fields"]["comment"]["comments"]:
                    content["comments"].append({
                        "author": comment["author"]["displayName"],
                        "body": comment["body"],
                        "created": comment["created"]
                    })
            else:
                debug_log("No comments found in ticket data")

            # Extract attachments
            attachment_count = 0
            if "attachment" in ticket_data["fields"]:
                attachment_count = len(ticket_data["fields"]["attachment"])
                debug_log(f"Found {attachment_count} attachments in ticket")
                
                for attachment in ticket_data["fields"]["attachment"]:
                    content["attachments"].append({
                        "filename": attachment["filename"],
                        "created": attachment["created"],
                        "url": attachment["content"]
                    })
            else:
                debug_log("No attachments found in ticket data")
                
            debug_log(f"Extracted content for ticket {content['key']} - {comment_count} comments, {attachment_count} attachments")
            return content
        except KeyError as e:
            error_log(f"KeyError when extracting ticket content: {e}")
            error_log(f"Available fields: {list(ticket_data['fields'].keys())}")
            raise

    def generate_summary(self, ticket_content: Dict) -> str:
        """Generate summary using Gemini"""
        debug_log(f"Generating summary for ticket: {ticket_content['key']}")
        
        prompt_text = f"""Summarize this Jira ticket with a focus on clarity and accuracy. The summary should cover:
1. Key issue details
2. A chronological timeline of customer interactions
3. Relevant technical discussions or proposed solutions
4. Current resolution status

Ticket Information:
Key: {ticket_content['key']}
Summary: {ticket_content['summary']}
Description: {ticket_content['description'][:500]}... (truncated)
Status: {ticket_content['status']}

Comments History:
{self._format_comments(ticket_content['comments'])}

Attachments:
{self._format_attachments(ticket_content['attachments'])}
"""
        debug_log(f"Prompt length: {len(prompt_text)} characters")
        
        prompt = {
            "contents": [{
                "parts": [{
                    "text": prompt_text
                }]
            }]
        }

        debug_log(f"Calling Gemini API for ticket summary")
        response = self.gemini_client.call_gemini_api(prompt)
        
        if response and 'candidates' in response:
            summary = response['candidates'][0]['content']['parts'][0]['text']
            debug_log(f"Successfully generated summary ({len(summary)} chars)")
            debug_log(f"First 200 chars of summary: {summary[:200]}...")
            return summary
        else:
            error_log(f"Failed to generate summary. Response: {json.dumps(response)}")
            if 'error' in response:
                error_log(f"Error details: {response['error']}")
            return "Failed to generate summary"

    def _format_comments(self, comments: List[Dict]) -> str:
        formatted = "\n".join([
            f"- {comment['created']}: {comment['author']}: {comment['body'][:100]}..." if len(comment['body']) > 100 else f"- {comment['created']}: {comment['author']}: {comment['body']}"
            for comment in comments
        ])
        debug_log(f"Formatted {len(comments)} comments, total size: {len(formatted)} chars")
        return formatted

    def _format_attachments(self, attachments: List[Dict]) -> str:
        formatted = "\n".join([
            f"- {attachment['created']}: {attachment['filename']}"
            for attachment in attachments
        ])
        debug_log(f"Formatted {len(attachments)} attachments, total size: {len(formatted)} chars")
        return formatted

    def add_summary_comment(self, issue_key: str, summary: str) -> bool:
        """Add the generated summary as an internal comment"""
        try:
            url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}/comment"
            payload = {
                "body": f"*AI-Generated Ticket Summary*\n\n{summary}",
                "properties": [{"key": "sd.public.comment", "value": {"internal": True}}]
            }
            debug_log(f"Adding summary comment to {issue_key} with URL: {url}")
            debug_log(f"Comment payload size: {len(json.dumps(payload))} characters")
            
            start_time = datetime.now()
            response = requests.post(url, auth=self.auth, json=payload)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            debug_log(f"Jira comment API request completed in {duration:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code >= 200 and response.status_code < 300:
                debug_log(f"Successfully added comment to {issue_key}")
                debug_log(f"Comment response: {response.text[:500]}...")
                return True
            else:
                error_log(f"Failed to add comment. Status code: {response.status_code}")
                error_log(f"Response body: {response.text}")
                response.raise_for_status()
                return False
        except Exception as e:
            error_log(f"Error adding summary comment: {e}")
            error_log(f"{traceback.format_exc()}")
            return False


def handle_webhook(issue_key, jira_base_url, jira_email, jira_api_token, gemini_api_key):
    """Process a single Jira ticket and add an AI-generated summary comment"""
    debug_log(f"Processing issue_key: {issue_key}")
    
    try:
        # Initialize clients
        debug_log(f"Initializing GeminiClient")
        gemini_client = GeminiClient(api_key=gemini_api_key)
        
        debug_log(f"Initializing JiraTicketAnalyzer")
        analyzer = JiraTicketAnalyzer(
            jira_base_url=jira_base_url,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            gemini_client=gemini_client
        )

        # Get ticket details
        debug_log(f"Getting ticket details for {issue_key}")
        ticket_data = analyzer.get_ticket_details(issue_key)
        if not ticket_data:
            error_log(f"Failed to fetch ticket details for {issue_key}")
            return {
                "success": False, 
                "error": f"Failed to fetch ticket details for {issue_key}"
            }

        # Extract relevant content
        debug_log(f"Extracting content from ticket data")
        ticket_content = analyzer.extract_ticket_content(ticket_data)

        # Generate summary
        debug_log(f"Generating summary")
        summary = analyzer.generate_summary(ticket_content)

        # Add summary as internal comment
        debug_log(f"Adding summary comment")
        comment_added = analyzer.add_summary_comment(issue_key, summary)
        
        if comment_added:
            debug_log(f"Successfully added comment to ticket {issue_key}")
        else:
            error_log(f"Failed to add comment to ticket {issue_key}")

        debug_log(f"Processing complete for {issue_key}")
        return {
            "success": True,
            "issue_key": issue_key,
            "summary_added": comment_added,
            "summary": summary
        }
    except Exception as e:
        error_log(f"Unexpected error in handle_webhook: {e}")
        error_log(f"{traceback.format_exc()}")
        return {
            "success": False, 
            "error": str(e)
        }

# Main function that will be called by Zapier
def main(zapier):
    debug_log("Starting Zapier Code step execution")
    
    # Get input data from the trigger or previous steps
    trigger_data = zapier.trigger_output
    debug_log(f"Received trigger data: {json.dumps(trigger_data, default=str)[:1000]}...")
    
    # Extract the issue key from trigger data
    issue_key = issue_key = zapier.trigger_output["issue"]["key"]
    if not issue_key:
        error_log("Missing required issue_key in trigger data")
        return {"success": False, "error": "Missing required issue_key in trigger data"}
    
    # Get credentials from environment variables
    try:
        # Get credentials from environment variables
        jira_base_url = os.environ.get("JIRA_BASE_URL")
        jira_email = os.environ.get("JIRA_EMAIL") 
        jira_api_token = os.environ.get("JIRA_API_TOKEN")
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        # Validate credentials
        missing = []
        if not jira_base_url: missing.append("JIRA_BASE_URL")
        if not jira_email: missing.append("JIRA_EMAIL")
        if not jira_api_token: missing.append("JIRA_API_TOKEN")
        if not gemini_api_key: missing.append("GEMINI_API_KEY")
        
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            error_log(error_msg)
            return {"success": False, "error": error_msg}
            
        debug_log(f"Retrieved credentials from environment variables")
        
        # Process the Jira ticket
        result = handle_webhook(
            issue_key=issue_key,
            jira_base_url=jira_base_url,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            gemini_api_key=gemini_api_key
        )
        
        debug_log(f"Completed processing with result: {json.dumps(result, default=str)}")
        return result
        
    except Exception as e:
        error_log(f"Error in main Zapier function: {e}")
        error_log(traceback.format_exc())
        return {"success": False, "error": str(e)}

output = main(zapier)
