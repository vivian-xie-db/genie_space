###
# IMPORTS AND CONFIGURATION
#
# This section imports necessary libraries, including pandas for data handling,
# time for polling delays, os and dotenv for environment variable management,
# and the Databricks SDK for interacting with the Genie API.
# It also configures basic logging.
###
import pandas as pd
import time
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")


###
# GenieClient CLASS
#
# This class acts as a client to interact with the Databricks Genie API. It
# encapsulates the methods required to start and manage conversations, send
# messages, and retrieve results, handling authentication and retry logic
# for API requests.
###
class GenieClient:
    ###
    # METHOD: __init__
    #
    # Initializes the GenieClient and configures the underlying Databricks
    # WorkspaceClient with specific credentials and robust retry settings for
    # network requests, including timeout, max retries, and exponential backoff.
    ###
    def __init__(self, host: str, space_id: str, token: str):
        self.host = host
        self.space_id = space_id
        self.token = token
        
        config = Config(
            host=f"https://{host}",
            token=token,
            auth_type="pat",
            retry_timeout_seconds=300,
            max_retries=5,
            retry_delay_seconds=2,
            retry_backoff_factor=2
        )
        self.client = WorkspaceClient(config=config)

    ###
    # METHOD: start_conversation
    #
    # Begins a new conversation in the specified Genie space with an initial
    # question and returns the new conversation and message IDs.
    ###
    def start_conversation(self, question: str) -> Dict[str, Any]:
        response = self.client.genie.start_conversation(
            space_id=self.space_id,
            content=question
        )
        return {
            "conversation_id": response.conversation_id,
            "message_id": response.message_id
        }

    ###
    # METHOD: send_message
    #
    # Sends a follow-up message to an already existing conversation and
    # returns the ID of the newly created message.
    ###
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        response = self.client.genie.create_message(
            space_id=self.space_id,
            conversation_id=conversation_id,
            content=message
        )
        return {
            "message_id": response.message_id
        }

    ###
    # METHOD: get_message
    #
    # Retrieves the full details and status of a specific message within
    # a conversation.
    ###
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        response = self.client.genie.get_message(
            space_id=self.space_id,
            conversation_id=conversation_id,
            message_id=message_id
        )
        return response.as_dict()

    ###
    # METHOD: get_query_result
    #
    # Fetches the results of a query associated with a message attachment. It
    # extracts the data and schema from the response.
    ###
    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        response = self.client.genie.get_message_attachment_query_result(
            space_id=self.space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            attachment_id=attachment_id
        )
        
        if hasattr(response, 'statement_response') and response.statement_response and \
           hasattr(response.statement_response, 'result') and response.statement_response.result:
            data_array = response.statement_response.result.data_array or []
        else:
            raise ValueError("Query execution failed: No result data available.")
            
        schema = {}
        if hasattr(response, 'statement_response') and response.statement_response and \
           hasattr(response.statement_response, 'manifest') and response.statement_response.manifest and \
           hasattr(response.statement_response.manifest, 'schema') and response.statement_response.manifest.schema:
            schema = response.statement_response.manifest.schema.as_dict()
            
        return {
            'data_array': data_array,
            'schema': schema
        }

    ###
    # METHOD: execute_query
    #
    # A wrapper method to execute a query using its attachment ID.
    ###
    def execute_query(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        response = self.client.genie.execute_query(
            space_id=self.space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            attachment_id=attachment_id
        )
        return response.as_dict()

    ###
    # METHOD: wait_for_message_completion
    #
    # Polls the `get_message` endpoint at regular intervals until the message
    # reaches a terminal state (e.g., COMPLETED, ERROR). This is a blocking
    # call that waits for Genie to finish processing.
    ###
    def wait_for_message_completion(self, conversation_id: str, message_id: str, timeout: int = 300, poll_interval: int = 2) -> Dict[str, Any]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            message = self.get_message(conversation_id, message_id)
            status = message.get("status")
            if status in ["COMPLETED", "ERROR", "FAILED"]:
                return message
            time.sleep(poll_interval)
        raise TimeoutError(f"Message processing timed out after {timeout} seconds")

    ###
    # METHOD: list_spaces
    #
    # Retrieves a list of all Genie spaces available to the user, handling
    # pagination to ensure all spaces are returned.
    ###
    def list_spaces(self) -> list:
        all_spaces = []
        next_page_token = None
        while True:
            response = self.client.genie.list_spaces(page_size=1000, page_token=next_page_token)
            if hasattr(response, 'spaces') and response.spaces:
                all_spaces.extend([space.as_dict() for space in response.spaces])
            next_page_token = getattr(response, 'next_page_token', None)
            if not next_page_token:
                break
        return all_spaces


###
# FUNCTION: process_genie_response
#
# Parses the completed message from Genie to extract the meaningful result.
# It handles different response types, such as plain text or a query result,
# converting query data into a pandas DataFrame.
###
def process_genie_response(client: GenieClient, conversation_id: str, message_id: str, complete_message: Dict[str, Any]) -> Tuple[Union[str, pd.DataFrame], Optional[str], Optional[str]]:
    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")
        
        if "text" in attachment and "content" in attachment["text"]:
            return attachment["text"]["content"], None, None
        
        elif "query" in attachment:
            query_info = attachment.get("query", {})
            query_text = query_info.get("query", "")
            description = query_info.get("description")
            
            query_result = client.get_query_result(conversation_id, message_id, attachment_id)
           
            data_array = query_result.get('data_array', [])
            schema = query_result.get('schema', {})
            columns = [col.get('name') for col in schema.get('columns', [])]
            
            if data_array:
                if not columns and data_array and data_array[0]:
                    columns = [f"column_{i}" for i in range(len(data_array[0]))]
                df = pd.DataFrame(data_array, columns=columns)
                return df, query_text, description
    
    if 'content' in complete_message:
        return complete_message.get('content', ''), None, None
    
    return "No response available", None, None


###
# FUNCTION: start_new_conversation
#
# Handles the logic for starting a new conversation. It uses the provided
# client to send the first message, wait for it to complete, and process
# the final response.
###
def start_new_conversation(client: GenieClient, question: str) -> Tuple[str, Union[str, pd.DataFrame], Optional[str], Optional[str]]:
    try:
        response = client.start_conversation(question)
        conversation_id = response["conversation_id"]
        message_id = response["message_id"]
        
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        result, query_text, description = process_genie_response(client, conversation_id, message_id, complete_message)
        
        return conversation_id, result, query_text, description
    except Exception as e:
        if "Expired Token" in str(e):
            return None, "Sorry, your authentication token has expired. Please refresh the page and try again.", None, None
        return None, f"Sorry, an error occurred: {str(e)}. Please try again.", None, None


###
# FUNCTION: continue_conversation
#
# Handles the logic for sending a message to an existing conversation. It
# uses the provided client to send the follow-up message, wait for completion,
# and process the final response.
###
def continue_conversation(client: GenieClient, conversation_id: str, question: str) -> Tuple[Union[str, pd.DataFrame], Optional[str], Optional[str]]:
    logger.info(f"Continuing conversation {conversation_id} with question: {question[:30]}...")
    try:
        response = client.send_message(conversation_id, question)
        message_id = response["message_id"]
        
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        result, query_text, description = process_genie_response(client, conversation_id, message_id, complete_message)
        
        return result, query_text, description
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            return "Sorry, the system is currently experiencing high demand. Please try again in a few moments.", None, None
        elif "Conversation not found" in str(e):
            return "Sorry, the previous conversation has expired. Please try your query again to start a new conversation.", None, None
        elif "Expired Token" in str(e):
            return "Sorry, your authentication token has expired. Please refresh the page and try again.", None, None
        else:
            logger.error(f"Error continuing conversation: {str(e)}")
            return f"Sorry, an error occurred: {str(e)}", None, None

###
# FUNCTION: genie_query
#
# This is the main entry point function for the application's backend logic.
# It instantiates the GenieClient once and then determines whether to start a
# new conversation or continue an existing one based on the presence of a `conversation_id`.
###
def genie_query(question: str, token: str, space_id: str, conversation_id: Optional[str] = None) -> Union[Tuple[str, Union[str, pd.DataFrame], Optional[str], Optional[str]], Tuple[None, str, None, None]]:
    try:
        client = GenieClient(host=DATABRICKS_HOST, space_id=space_id, token=token)
        
        if conversation_id:
            result, query_text, description = continue_conversation(client, conversation_id, question)
        else:
            conversation_id, result, query_text, description = start_new_conversation(client, question)
        
        return conversation_id, result, query_text, description
            
    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}. Please try again.")
        return None, f"Sorry, an error occurred: {str(e)}. Please try again.", None, None