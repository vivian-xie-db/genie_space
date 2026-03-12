import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
import backoff
import uuid
from token_minter import TokenMinter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load environment variables
SPACE_ID = os.environ.get("SPACE_ID")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")

token_minter = TokenMinter(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    host=DATABRICKS_HOST
)


@dataclass
class GenieResponse:
    """Structured response from Genie containing all available data."""
    text_response: Optional[str] = None
    sql_query: Optional[str] = None
    sql_description: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    data_summary: Optional[str] = None
    status: str = "OK"
    error: Optional[str] = None


class GenieClient:
    def __init__(self, host: str, space_id: str):
        self.host = host
        self.space_id = space_id
        self.update_headers()
        
        self.base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
    
    def update_headers(self) -> None:
        """Update headers with fresh token from token_minter"""
        self.headers = {
            "Authorization": f"Bearer {token_minter.get_token()}",
            "Content-Type": "application/json"
        }
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def start_conversation(self, question: str) -> Dict[str, Any]:
        """Start a new conversation with the given question"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/start-conversation"
        payload = {"content": question}
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Send a follow-up message to an existing conversation"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": message}
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        """Get the details of a specific message"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Get the query result using the attachment_id endpoint"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        # Extract data_array from the correct nested location
        data_array = []
        if 'statement_response' in result:
            if 'result' in result['statement_response']:
                data_array = result['statement_response']['result'].get('data_array', [])
            
        return {
                    'data_array': data_array,
                    'schema': result.get('statement_response', {}).get('manifest', {}).get('schema', {})
                }

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def execute_query(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Execute a query using the attachment_id endpoint"""
        self.update_headers()  # Refresh token before API call
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/execute-query"
        
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    

    def wait_for_message_completion(self, conversation_id: str, message_id: str, timeout: int = 300, poll_interval: int = 2) -> Dict[str, Any]:
        """
        Wait for a message to reach a terminal state (COMPLETED, ERROR, etc.).
        
        Args:
            conversation_id: The ID of the conversation
            message_id: The ID of the message
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
            
        Returns:
            The completed message
        """
        
        start_time = time.time()
        attempt = 1
        
        while time.time() - start_time < timeout:
            
            message = self.get_message(conversation_id, message_id)
            status = message.get("status")
            
            if status in ["COMPLETED", "ERROR", "FAILED"]:
                return message
                
            time.sleep(poll_interval)
            attempt += 1
            
        raise TimeoutError(f"Message processing timed out after {timeout} seconds")

def start_new_conversation(question: str) -> Tuple[Optional[str], GenieResponse]:
    """
    Start a new conversation with Genie.

    Args:
        question: The initial question

    Returns:
        Tuple of (conversation_id, GenieResponse)
    """
    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID
    )

    try:
        response = client.start_conversation(question)
        conversation_id = response.get("conversation_id")
        message_id = response.get("message_id")

        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        result = process_genie_response(client, conversation_id, message_id, complete_message)

        return conversation_id, result

    except Exception as e:
        return None, GenieResponse(
            text_response=f"Sorry, an error occurred: {str(e)}. Please try again.",
            status="ERROR",
            error=str(e)
        )

def continue_conversation(conversation_id: str, question: str) -> GenieResponse:
    """
    Send a follow-up message in an existing conversation.

    Args:
        conversation_id: The existing conversation ID
        question: The follow-up question

    Returns:
        GenieResponse with all available data
    """
    logger.info(f"Continuing conversation {conversation_id} with question: {question[:30]}...")

    client = GenieClient(
        host=DATABRICKS_HOST,
        space_id=SPACE_ID
    )

    try:
        response = client.send_message(conversation_id, question)
        message_id = response.get("message_id")

        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        return process_genie_response(client, conversation_id, message_id, complete_message)

    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            msg = "Sorry, the system is currently experiencing high demand. Please try again in a few moments."
        elif "Conversation not found" in str(e):
            msg = "Sorry, the previous conversation has expired. Please try your query again to start a new conversation."
        else:
            logger.error(f"Error continuing conversation: {str(e)}")
            msg = f"Sorry, an error occurred: {str(e)}"
        return GenieResponse(text_response=msg, status="ERROR", error=str(e))

def _generate_data_summary(df: pd.DataFrame) -> str:
    """Generate a brief summary of a DataFrame's contents."""
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}"]

    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols[:5]:
        lines.append(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

    if len(numeric_cols) > 5:
        lines.append(f"  ... and {len(numeric_cols) - 5} more numeric columns")

    return "\n".join(lines)


def process_genie_response(client, conversation_id, message_id, complete_message) -> GenieResponse:
    """
    Process the response from Genie, collecting ALL available data.

    Args:
        client: The GenieClient instance
        conversation_id: The conversation ID
        message_id: The message ID
        complete_message: The completed message response

    Returns:
        GenieResponse with all available fields populated
    """
    response = GenieResponse()

    # Collect text from message content
    if 'content' in complete_message:
        response.text_response = complete_message.get('content', '')

    # Process all attachments (don't return early)
    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")

        # Collect text content from attachment
        if "text" in attachment and "content" in attachment["text"]:
            response.text_response = attachment["text"]["content"]

        # Collect query and data from attachment
        if "query" in attachment:
            query_info = attachment.get("query", {})
            response.sql_query = query_info.get("query", "")
            response.sql_description = query_info.get("description", None)

            try:
                query_result = client.get_query_result(conversation_id, message_id, attachment_id)

                data_array = query_result.get('data_array', [])
                schema = query_result.get('schema', {})
                columns = [col.get('name') for col in schema.get('columns', [])]

                if data_array:
                    if not columns and len(data_array) > 0:
                        columns = [f"column_{i}" for i in range(len(data_array[0]))]

                    df = pd.DataFrame(data_array, columns=columns)

                    # Try to convert numeric columns
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except (ValueError, TypeError):
                            pass

                    response.data = df
                    response.data_summary = _generate_data_summary(df)
            except Exception as e:
                logger.warning(f"Failed to get query result: {e}")

    # If nothing was populated, set a default text
    if response.text_response is None and response.data is None:
        response.text_response = "No response available"

    return response

def genie_query(question: str) -> GenieResponse:
    """
    Main entry point for querying Genie.

    Args:
        question: The question to ask

    Returns:
        GenieResponse with text_response, sql_query, sql_description, data, and data_summary
    """
    try:
        conversation_id, result = start_new_conversation(question)
        return result

    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}. Please try again.")
        return GenieResponse(
            text_response=f"Sorry, an error occurred: {str(e)}. Please try again.",
            status="ERROR",
            error=str(e)
        )

