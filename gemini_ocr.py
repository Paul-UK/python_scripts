import requests
import base64
import json
import re

class GeminiOcrTableFigureParser:
    """
    OCRs an image for tables, figures, and drawings using Gemini Pro API.
    Expects bounding boxes (bbox) in xyxy format and extracts text content for each.
    If a bbox corresponds to a table, all cells are also extracted as individual bboxes.
    """
    def __init__(self, gemini_api_key: str, jira_token=None):
        self.api_key = gemini_api_key
        self.jira_token = jira_token
        self.model = "gemini-2.5-pro"  # Use a stable model name
        self.base_api_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.REQUEST_TIMEOUT = 60  # seconds
        self.temperature = 0.1  # lower temperature for most deterministic responses
        self.max_tokens = 8192  # Increased token limit for longer responses
        # top_p is irrelevant if top_k is 1, but set to 1.0 to be safe
        self.top_p = 1.0
        self.top_k = 1
        
        # Build the full API URL
        self.gemini_url = f"{self.base_api_url}{self.model}:generateContent"

    def build_prompt(self):
        """
        Returns a placeholder prompt for Gemini API to extract bboxes and text.
        """
        return (
            "You are an excellent document parser. Detect and return all tables, figures, and drawings in the image. "
            "For each, extract their bounding box coordinates as [x1, y1, x2, y2] (top-left and bottom-right corners), and the contained text. "
            "If it is a table, detect all cell bboxes with text, and provide their bboxes and texts as well. "
            "Return a JSON list with structure: "
            "[{'type': 'table'|'figure'|'drawing', 'bbox': [x1, y1, x2, y2], 'text': '...', 'cells': [{'bbox': [x1, y1, x2, y2], 'text': '...'}]}]"
        )

    def query_gemini_ocr(self, image_url: str) -> dict:
        """
        Calls the Gemini Pro Vision API for OCR and returns LLM results.
        """
        # Download and encode image properly
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_base64 = base64.b64encode(image_response.content).decode('utf-8')
        
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": self.build_prompt()},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "topK": self.top_k,
                "maxOutputTokens": self.max_tokens
            }
        }
        
        params = {'key': self.api_key}
        response = requests.post(
            self.gemini_url, 
            headers=headers, 
            params=params, 
            json=body,
            timeout=self.REQUEST_TIMEOUT
        )
        
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            response.raise_for_status()
            
        result = response.json()
        
        # Check if response was truncated
        if result.get('candidates') and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            finish_reason = candidate.get('finishReason')
            
            if finish_reason == 'MAX_TOKENS':
                raise ValueError("Response was truncated due to max tokens limit. Try reducing image size or increasing max_tokens.")
            elif finish_reason == 'SAFETY':
                raise ValueError("Response blocked by safety filters.")
            elif finish_reason not in ['STOP', None]:
                raise ValueError(f"Unexpected finish reason: {finish_reason}")
        
        # Debug: Print the response structure to understand it
        print("API Response structure:", json.dumps(result, indent=2))
        
        # Try to extract text from various possible response structures
        text_result = None
        
        try:
            # Standard structure
            text_result = result['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            try:
                # Alternative structure 1
                text_result = result['candidates'][0]['content']['text']
            except (KeyError, IndexError):
                try:
                    # Alternative structure 2
                    text_result = result['candidates'][0]['text']
                except (KeyError, IndexError):
                    try:
                        # Alternative structure 3
                        text_result = result['text']
                    except (KeyError, IndexError):
                        # Check if content exists but is empty due to truncation
                        if 'candidates' in result and result['candidates']:
                            content = result['candidates'][0].get('content', {})
                            if content.get('role') == 'model' and 'parts' not in content:
                                raise ValueError("Response content is empty - likely due to truncation or filtering")
                        raise ValueError(f"Could not extract text from API response. Response structure: {result}")
        
        if not text_result:
            raise ValueError("Empty text result from API")
            
        print(f"Extracted text: {text_result}")
        
        # Find JSON block in text_result using regex (to be robust with LLMs)
        json_match = re.search(r"(\[.*\])", text_result, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not extract JSON response from LLM output. Text was: {text_result}")
        
        try:
            objects = json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}. Extracted JSON: {json_match.group(1)}")
            
        return objects

    def to_labelbox_bbox_import(self, objects, global_image_width, global_image_height):
        """
        Converts OCRed objects (including any cells) to Labelbox bbox import format.
        Each bbox becomes a Labelbox annotation with schema for 'bbox' and associated text.
        See https://github.com/Labelbox/labelbox-python/blob/develop/examples/annotation_import/image.ipynb
        """
        import uuid
        labelbox_annotations = []
        # Examples expect coordinates as [left, top, width, height]
        def xyxy_to_ltwh(bbox):
            x1, y1, x2, y2 = bbox
            return [x1, y1, x2 - x1, y2 - y1]
        
        for obj in objects:
            # Ensure all required keys are present
            if "bbox" not in obj or "text" not in obj:
                continue
              #--- Placeholder ---
            annotation = {
                "uuid": str(uuid.uuid4()),
                "schemaId": "YOUR_BBOX_SCHEMA_ID", 
                "dataRow": {"globalKey": "YOUR_IMAGEROW_GLOBAL_KEY"},
                "bbox": {
                    "left": obj["bbox"][0],
                    "top": obj["bbox"][1],
                    "width": obj["bbox"][2] - obj["bbox"][0],
                    "height": obj["bbox"][3] - obj["bbox"][1]
                },
                "classifications": [
                    {
                        "schemaId": "YOUR_TEXT_SCHEMA_ID", # replace with schema id for free-text
                        "answer": obj.get("text", "")
                    }
                ]
            }
            annotation["bbox"]["left"] = max(0, min(annotation["bbox"]["left"], global_image_width))
            annotation["bbox"]["top"] = max(0, min(annotation["bbox"]["top"], global_image_height))
            annotation["bbox"]["width"] = max(1, min(annotation["bbox"]["width"], global_image_width - annotation["bbox"]["left"]))
            annotation["bbox"]["height"] = max(1, min(annotation["bbox"]["height"], global_image_height - annotation["bbox"]["top"]))
            labelbox_annotations.append(annotation)

            # Add table cell bboxes as separate annotations if present
            if obj.get("type") == "table" and "cells" in obj:
                for cell in obj["cells"]:
                    if "bbox" in cell and "text" in cell:
                        cell_ann = {
                            "uuid": str(uuid.uuid4()),
                            "schemaId": "YOUR_BBOX_SCHEMA_ID",  # same or different as above
                            "dataRow": {"globalKey": "YOUR_IMAGEROW_GLOBAL_KEY"},  # as above
                            "bbox": {
                                "left": cell["bbox"][0],
                                "top": cell["bbox"][1],
                                "width": cell["bbox"][2] - cell["bbox"][0],
                                "height": cell["bbox"][3] - cell["bbox"][1]
                            },
                            "classifications": [
                                {
                                    "schemaId": "YOUR_TEXT_SCHEMA_ID",  # as above
                                    "answer": cell.get("text", "")
                                }
                            ]
                        }
                        cell_ann["bbox"]["left"] = max(0, min(cell_ann["bbox"]["left"], global_image_width))
                        cell_ann["bbox"]["top"] = max(0, min(cell_ann["bbox"]["top"], global_image_height))
                        cell_ann["bbox"]["width"] = max(1, min(cell_ann["bbox"]["width"], global_image_width - cell_ann["bbox"]["left"]))
                        cell_ann["bbox"]["height"] = max(1, min(cell_ann["bbox"]["height"], global_image_height - cell_ann["bbox"]["top"]))
                        labelbox_annotations.append(cell_ann)
        
        return labelbox_annotations

def run_gemini_ocr_pipeline(image_url, gemini_api_key, image_width, image_height):
    """
    Complete example pipeline for Gemini OCR to Labelbox format.
    """
    parser = GeminiOcrTableFigureParser(gemini_api_key)
    objects = parser.query_gemini_ocr(image_url)
    labelbox_format = parser.to_labelbox_bbox_import(objects, image_width, image_height)
    return labelbox_format

# USAGE (Pseudo):
labelbox_data = run_gemini_ocr_pipeline(
     image_url="",
     gemini_api_key="",
     image_width=1700,
     image_height=2200
 )
print(labelbox_data)
