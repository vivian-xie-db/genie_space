###
# IMPORTS AND INITIAL SETUP
#
# This section imports all necessary libraries for the Dash application,
# including Dash itself, Dash Bootstrap Components for styling, pandas for
# data manipulation, and other utilities for handling environment variables,
# API calls, and logging.
###
import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context, no_update, clientside_callback, dash_table, DiskcacheManager
import dash_bootstrap_components as dbc
import json
import pandas as pd
import os
import uuid
import sqlparse
import logging
from flask import request
from io import StringIO
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.config import Config
from databricks.sdk.errors import DatabricksError
from genie_room import genie_query, GenieClient
from flask_caching import Cache
import diskcache

# Load environment variables from a .env file for configuration.
load_dotenv()

# Configure the logging system.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###
# SERVER-SIDE CACHE
#
# This dictionary serves as a simple in-memory cache for storing pandas
# DataFrames generated during user sessions. Instead of sending large DataFrames
# to the client via dcc.Store, we store them here and reference them by a
# unique UUID. This significantly reduces network traffic and improves app performance.
# NOTE: For production environments with multiple workers, a more robust caching
# solution like Redis or Flask-Caching would be recommended to handle state
# correctly and prevent memory issues.

###
# DASH APPLICATION INITIALIZATION
#
# The main Dash application instance is created here, with external stylesheets
# for Bootstrap theming and a title for the browser tab.
###

# Initialize diskcache for long callbacks
# This will store background callback results in a 'cache' directory
cache_disk = diskcache.Cache("./cache")
long_callback_manager = DiskcacheManager(cache_disk)

# Initialize a separate diskcache for DataFrames to be shared across processes
# This will be used instead of the in-memory DATAFRAME_CACHE dictionary
df_cache_for_long_callbacks = diskcache.Cache("./dataframe_cache")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="BI Agent",
    background_callback_manager=long_callback_manager
)

# Initialize Flask-Caching
# For production, consider using a more robust backend like Redis.
# Example: {'CACHE_TYPE': 'redis', 'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379/0')}
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache', # Use SimpleCache for in-memory caching
    'CACHE_DEFAULT_TIMEOUT': 300 # Cache items expire after 300 seconds (5 minutes)
})

# Initialize WorkspaceClient globally
# This ensures the client is created once and reused across requests.
# It uses environment variables DATABRICKS_HOST and DATABRICKS_TOKEN by default.
try:
    global_workspace_client = WorkspaceClient()
    logger.info("Databricks WorkspaceClient initialized globally.")
except Exception as e:
    logger.error(f"Failed to initialize Databricks WorkspaceClient globally: {e}")
    global_workspace_client = None

###
# APPLICATION LAYOUT DEFINITION
#
# The `app.layout` defines the complete HTML structure of the single-page
# application. It includes dcc.Store components for state management, a top
# navigation bar, a main content area for the chat interface, and a modal
# for generating insights. The layout is designed to be dynamic, with
# components shown or hidden based on user interaction.
###
app.layout = html.Div([
    html.Div([
        # Stores for holding client-side data and state without displaying it.
        dcc.Store(id="selected-space-id", data=None, storage_type='session'),
        dcc.Store(id="spaces-list", data=[]),
        dcc.Store(id="conversation-id-store", data=None, storage_type='session'),
        dcc.Store(id="username-store", data=None),
        dcc.Store(id="current-dataframe-uuid", data=None),
        dcc.Store(id="processed-export-clicks", data={}, storage_type='session'),
        dcc.Store(id="processed-insight-clicks", data={}, storage_type='session'),
        dcc.Store(id="processed-confirm-insight-clicks", data={}, storage_type='session'),
        dcc.Store(id="insight-trigger-store", data=None),
        dcc.Store(id="user-token-store", data=None),
        
        # Top navigation bar, fixed to the top of the screen.
        html.Div([
            html.Div([
                html.Div([
                    html.Button(html.Img(src="assets/menu_icon.svg", className="menu-icon"), id="sidebar-toggle", className="nav-button"),
                    html.Button(html.Img(src="assets/plus_icon.svg", className="new-chat-icon"), id="new-chat-button", className="nav-button", disabled=False, title="New chat"),
                    html.Button([html.Img(src="assets/plus_icon.svg", className="new-chat-icon"), html.Div("New chat", className="new-chat-text")], id="sidebar-new-chat-button", className="new-chat-button", disabled=False),
                    html.Button(html.Img(src="assets/change.png", style={'height': '16px'}), id="change-space-button", className="nav-button", disabled=False, title="Change Agent"),
                ], id="nav-left", className="nav-left"),
                html.Div([
                    html.Div([html.Div(className="sidebar-header-text", children="Your recent conversations")], className="sidebar-header"),
                    html.Div([], className="chat-list", id="chat-list")
                ], id="sidebar", className="sidebar")
            ], id="left-component", className="left-component"),
            html.Div(html.Div(html.Div(className="company-logo-black"), id="logo-container", className="logo-container"), id="nav-center", className="nav-center"),
            html.Div(dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem(id="user-email-display", style={'color': 'black', 'fontSize': '14px'}, disabled=True),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Logout", href=f"https://{os.getenv('DATABRICKS_HOST')}/login.html", external_link=True, style={'color': 'black', 'fontSize': '14px'}, className="logout-link"),
                ],
                label=html.Div(id="user-avatar-initials", className="user-avatar"),
                nav=True, in_navbar=True, align_end=True, toggle_style={"background": "transparent", "border": "none", "padding": "0"}, className="user-dropdown"
            ), id="nav-right", className="nav-right"),
        ], id="top-nav", className="top-nav", style={"display": "none"}),
        
        # Main content wrapper for the agent selection overlay and chat interface.
        html.Div([
            # Agent selection overlay. This is the first screen the user sees.
            html.Div([
                html.Div([
                    html.Div(className="company-logo"),
                    html.Div(dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem(id="user-email-display-overlay", style={'color': 'black', 'fontSize': '14px'}, disabled=True),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Logout", href=f"https://{os.getenv('DATABRICKS_HOST')}/login.html", external_link=True, style={'color': 'black', 'fontSize': '14px'}, className="logout-link"),
                        ],
                        label=html.Div(id="user-avatar-initials-overlay", className="user-avatar"),
                        nav=True, in_navbar=True, align_end=True, toggle_style={"background": "transparent", "border": "none", "padding": "0"}, className="user-dropdown"
                    ), className="nav-right"),
                    html.Div("BI Agent Platform", className="main-title"),
                    html.Div("Empowering insights through conversation", className="space-select-tagline"),
                    html.Div(id="welcome-user-greeting", className="greeting-title"),
                    html.Div([
                        dcc.Dropdown(id="space-dropdown", options=[], placeholder="Choose an Agent", className="space-select-dropdown", optionHeight=40, searchable=True, style={'z-index': 1000}),
                        html.Button("Explore Agent", id="select-space-button", className="space-select-button", disabled=True)
                    ], className="explore-agent-container"),
                    html.Div([
                        html.A(html.Button("Documentation", className="request-support-button"), href="https://jda365.sharepoint.com/:b:/r/sites/O365-AnalyticsJDA/Shared%20Documents/Cognitive%20Analytics/BI%20Agent%20Documentation.pdf?csf=1&web=1&e=7xdXIe", target="_blank"),
                        html.Span("|", style={'color': 'white', 'fontSize': '1em', 'padding': '0 0.5em'}),
                        html.A(html.Button("Request Support", className="request-support-button"), href="https://jdaswin.service-now.com/jdasp/?id=sc_cat_item&sys_id=997bc9dc1baea590766441dce54bcb6e&sysparm_category=afc4a6211b6aa1d0766441dce54bcbd6", target="_blank")
                    ], className="support-links"),
                    html.Div(id="space-select-error", className="space-select-error")
                ], className="space-select-card")
            ], id="space-select-container", className="space-select-container", style={"height": "100%", "top": "0"}),
            
            # Main chat interface area.
            html.Div([
                html.Div([
                    html.Div([
                        # Welcome screen with suggestions, visible for new chats.
                        html.Div([
                            html.Div(html.Div(html.Div(className="genie-logo"), className="genie-logo-container"), className="genie-logo-container-header"),
                            html.Div(html.Div(id="welcome-title", className="welcome-message"), className="welcome-title-container"),
                            html.Div(id="welcome-description", className="welcome-message-description"),
                            html.Div([
                                html.Button([html.Div(className="suggestion-icon"), html.Div("What is the purpose of this Agent? Give me a short summary.", className="suggestion-text", id="suggestion-1-text")], id="suggestion-1", className="suggestion-button"),
                                html.Button([html.Div(className="suggestion-icon"), html.Div("How to converse with the Agent? Give me an example prompt.", className="suggestion-text", id="suggestion-2-text")], id="suggestion-2", className="suggestion-button"),
                                html.Button([html.Div(className="suggestion-icon"), html.Div("Explain the dataset behind this Agent.", className="suggestion-text", id="suggestion-3-text")], id="suggestion-3", className="suggestion-button"),
                                html.Button([html.Div(className="suggestion-icon"), html.Div("What columns or fields are available in this dataset?", className="suggestion-text", id="suggestion-4-text")], id="suggestion-4", className="suggestion-button")
                            ], className="suggestion-buttons")
                        ], id="welcome-container", className="welcome-container visible"),
                        # Container where chat messages will be dynamically added.
                        html.Div([], id="chat-messages", className="chat-messages"),
                    ], id="chat-content", className="chat-content"),
                    # Fixed input area at the bottom of the chat.
                    html.Div([
                        html.Div([
                            dcc.Input(id="chat-input-fixed", placeholder="Ask your question...", className="chat-input", type="text", disabled=False),
                            html.Div([
                                html.Button(html.Img(src="assets/mic_icon.svg", style={'width': '18px', 'height': '18px'}), id="mic-button", className="input-button", title="Start/Stop Dictation"),
                                html.Button(id="send-button-fixed", className="input-button send-button", disabled=False)
                            ], className="input-buttons-right"),
                            html.Div("You can only submit one query at a time", id="query-tooltip", className="query-tooltip")
                        ], id="fixed-input-container", className="fixed-input-container"),
                        html.Div("Always review the accuracy of responses.", className="disclaimer-fixed")
                    ], id="fixed-input-wrapper", className="fixed-input-wrapper"),
                ], id="chat-container", className="chat-container"),
            ], id="main-content", className="main-content", style={"display": "none"}),
        ], style={"height": "100vh", "position": "relative"}),
        
        # Dummy divs and stores for facilitating callbacks.
        html.Div(id='dummy-output'),
        dcc.Store(id="chat-trigger", data={"trigger": False, "message": ""}),
        dcc.Store(id="chat-history-store", data=[], storage_type='session'),
        dcc.Store(id="query-running-store", data=False),
        dcc.Store(id="session-store", data={"current_session": None}, storage_type='session'),
        html.Div(id='dummy-insight-scroll'),
        dcc.Download(id="download-dataframe-csv"),
        
        # Modal dialog for generating insights from a table.
        dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("AI Analysis"), close_button=True, className="bg-white border-bottom"),
        dbc.ModalBody([
                html.P("Edit AI prompt:", className="mb-1 text-muted"),
                dcc.Textarea(id="insight-prompt-textarea",
                    value="As a data analyst, analyze the table data for key insights, patterns, and business implications. Be concise and actionable.",
                    style={"width": "100%", "height": "150px", "resize": "vertical", "borderRadius": "6px", "border": "1px solid #dee2e6", "padding": "8px"},
                    className="form-control"),
                html.Div(id="insight-generation-status", className="mt-2"),
            ],className="p-3"),
        dbc.ModalFooter(dbc.Button("Confirm", id="confirm-insight-prompt-button", color="primary", n_clicks=0, className="px-3 py-1"), className="border-top p-1"),
        ], id="insight-prompt-modal", is_open=False, size="md", centered=True, backdrop="static", style={"borderRadius": "6px"}
        ),
    ], id="app-inner-layout"),
], id="root-container")

###
# HELPER FUNCTIONS
#
# These utility functions are used within the callbacks to perform specific tasks
# like formatting SQL code or calling a Large Language Model (LLM).
###

@cache.memoize() # Cache the results of this function
def call_llm_for_insights(df_csv, prompt=None): # df is now passed as CSV string for caching
     """
     Sends a DataFrame (as CSV string) and a prompt to a Databricks Serving Endpoint to generate
     data insights using an LLM. This function now uses the streaming API.

     Args:
         df_csv (str): The data table as a CSV string to be analyzed.
         prompt (str, optional): A custom prompt for the LLM. A default is used if not provided.

     Returns:
         str: The complete text response generated by the LLM.
     """
     formatting_instruction = "\n\nIMPORTANT: Do not use markdown headers (e.g., '#', '##'). Instead, use bolding for titles (e.g., '**Key Insights**')."
     default_prompt = (
         "You are a professional data analyst. Given the following table data, provide deep, actionable analysis for\n"
         "1. Key insights and trends.\n"
         "2. Notable patterns\n"
         "3. Business implications.\n"
         "Be thorough, professional, and concise."
     )

     final_prompt = (prompt or default_prompt) + formatting_instruction
     full_prompt = f"{final_prompt}\n\nTable data:\n{df_csv}"

     try:
        # Use the globally initialized WorkspaceClient
        if global_workspace_client:
            response = global_workspace_client.serving_endpoints.query(
                os.getenv("SERVING_ENDPOINT_NAME"),
                messages=[ChatMessage(content=full_prompt, role=ChatMessageRole.USER)],
            )
            return response.choices[0].message.content
        else:
            return "Error: WorkspaceClient not initialized."

     except DatabricksError as dbe:
         logger.error(f"Databricks API Error generating insights: {dbe}")
         # Provide a more user-friendly message for specific, common errors
         if "PERMISSION_DENIED" in str(dbe):
             return "Error: You do not have permission to access the analysis model. Please contact support."
         # Attempt to log the raw response body if available in the DatabricksError
         if hasattr(dbe, 'body') and dbe.body:
             logger.error(f"DatabricksError response body: {dbe.body}")
         return f"Error communicating with the analysis service: {dbe.message}"
     except Exception as e:
         logger.error(f"An unexpected error occurred while generating insights: {str(e)}")
         return f"An unexpected error occurred while generating insights: {str(e)}"

###
# CALLBACKS
#
# This section contains all the Dash callbacks that provide interactivity to
# the application. Callbacks are Python functions decorated with `@app.callback`
# that are automatically called by Dash whenever an input component's property
# changes.
###

###
# CALLBACK: Handle User Input and Display "Thinking" Indicator
#
# This callback is the primary entry point for any user query, whether from a
# suggestion button or the text input. It updates the chat display with the
# user's message and a "Thinking..." indicator, then triggers the next callback
# in the chain to fetch the actual response. It also manages chat history and session state.
###
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-input-fixed", "value", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input("suggestion-1", "n_clicks"),
     Input("suggestion-2", "n_clicks"),
     Input("suggestion-3", "n_clicks"),
     Input("suggestion-4", "n_clicks"),
     Input("send-button-fixed", "n_clicks"),
     Input("chat-input-fixed", "n_submit")],
    [State("suggestion-1-text", "children"),
     State("suggestion-2-text", "children"),
     State("suggestion-3-text", "children"),
     State("suggestion-4-text", "children"),
     State("chat-input-fixed", "value"),
     State("chat-messages", "children"),
     State("welcome-container", "className"),
     State("chat-list", "children"),
     State("chat-history-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def handle_all_inputs(s1_clicks, s2_clicks, s3_clicks, s4_clicks, send_clicks, submit_clicks,
                     s1_text, s2_text, s3_text, s4_text, input_value, current_messages,
                     welcome_class, current_chat_list, chat_history, session_data):
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * 8

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    suggestion_map = {
        "suggestion-1": s1_text, "suggestion-2": s2_text,
        "suggestion-3": s3_text, "suggestion-4": s4_text
    }
    user_input = suggestion_map.get(trigger_id, input_value)

    if not user_input:
        return [no_update] * 8

    user_message = html.Div([
        html.Div(user_input, className="message-text"),
        html.Div(html.Div(id="user-avatar-initials", className="user-avatar-chat"), className="user-info")
    ], className="user-message message")

    updated_messages = (current_messages or []) + [user_message]

    thinking_indicator = html.Div(
        html.Div([html.Span(className="spinner"), html.Span("Thinking...")], className="thinking-indicator"),
        className="bot-message message"
    )
    updated_messages.append(thinking_indicator)

    chat_history = chat_history or []
    # Check if this is the start of a new session
    if session_data.get("current_session") is None:
        current_session_index = 0
        session_data = {"current_session": 0}
        # Create and insert the new session at the beginning of the history
        new_session = {
            "session_id": current_session_index,
            "queries": [user_input],
            "messages": updated_messages,
            "conversation_id": None
        }
        chat_history.insert(0, new_session)
    else:
        # Update an existing session
        current_session_index = session_data["current_session"]
        if current_session_index < len(chat_history):
            chat_history[current_session_index]["messages"] = updated_messages
            chat_history[current_session_index]["queries"].append(user_input)
        else:
            # Fallback for an invalid session index - treat as new
            current_session_index = 0
            session_data = {"current_session": 0}
            new_session = {
                "session_id": current_session_index,
                "queries": [user_input],
                "messages": updated_messages,
                "conversation_id": None
            }
            chat_history.insert(0, new_session)

    updated_chat_list = [
        html.Div(session["queries"][0], className=f"chat-item{' active' if i == current_session_index else ''}", id={"type": "chat-item", "index": i})
        for i, session in enumerate(chat_history)
    ]

    return (updated_messages, "", "welcome-container hidden",
            {"trigger": True, "message": user_input}, True,
            updated_chat_list, chat_history, session_data)

###
# CALLBACK: Fetch Backend Response (Converted to long_callback)
#
# Triggered by the `handle_all_inputs` callback, this function sends the user's
# query to the `genie_query` backend. It processes the response, which can be
# text or a DataFrame. If a DataFrame is returned, it's stored in the server-side
# cache and displayed as a DataTable. The final response replaces the
# "Thinking..." indicator in the chat.
###
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("conversation-id-store", "data", allow_duplicate=True)],
    Input("chat-trigger", "data"),
    [State("chat-messages", "children"),
     State("chat-history-store", "data"),
     State("selected-space-id", "data"),
     State("conversation-id-store", "data"),
     State("user-token-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True,
    background=True,
    # Define outputs to be updated while the callback is running
    running=[
        (Output("chat-input-fixed", "disabled"), True, False),
        (Output("mic-button", "disabled"), True, False),
        (Output("send-button-fixed", "disabled"), True, False),
        (Output("new-chat-button", "disabled"), True, False),
        (Output("sidebar-new-chat-button", "disabled"), True, False),
    ],
)
def get_model_response(trigger_data, current_messages, chat_history, selected_space_id, conversation_id, user_token, session_data):
    # This callback can now be long-running without blocking the main Dash thread.
    # The `running` argument will manage the disabled state of buttons.

    if not trigger_data or not trigger_data.get("trigger"):
        return no_update, no_update, no_update, no_update, no_update

    user_input = trigger_data.get("message", "")
    if not user_input:
        return no_update, no_update, no_update, no_update, no_update

    new_conv_id = conversation_id
    try:
        # Use the user_token passed as a State
        new_conv_id, response, query_text, description = genie_query(user_input, user_token, selected_space_id, conversation_id)

        content = None

        # Case 1: The response is a string.
        if isinstance(response, str):
            processed_response = response.replace('[', '\\[').replace(']', '\\]')
            content = dcc.Markdown(processed_response, className="message-text-bot")
        
        # Case 2: The response is a non-empty pandas DataFrame.
        elif isinstance(response, pd.DataFrame) and not response.empty:
            df_response = response  # Use the DataFrame directly
            
            # Case 2a: The DataFrame contains only a single value.
            if df_response.shape == (1, 1):
                content = dcc.Markdown(str(df_response.iloc[0, 0]))
            
            # Case 2b: The DataFrame is a full table.
            else:
                table_uuid = str(uuid.uuid4())
                # Store the DataFrame as a CSV string in the diskcache
                df_csv_string = df_response.to_csv(index=False)
                df_cache_for_long_callbacks.set(table_uuid, df_csv_string) # Use the new df_cache

                table_data = df_response.to_dict('records')
                table_columns = [{"name": col, "id": col} for col in df_response.columns]
                tooltip_data = [{col: {'value': str(row[col]), 'type': 'markdown'} for col in df_response.columns} for row in table_data]
                
                data_table = dash_table.DataTable(
                    id=f"table-{len(chat_history)}", data=table_data, columns=table_columns,
                    sort_action="native", filter_action="native",
                    style_table={'maxHeight': '300px', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '95%'},
                    style_data={'textAlign': 'left', 'padding': '5px', 'height': '40px', 'maxHeight': '40px', 'lineHeight': '14px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap', 'verticalAlign': 'top'},
                    style_header={'fontWeight': 'bold', 'textAlign': 'center', 'backgroundColor': '#f8f8f8', 'height': '40px', 'maxHeight': '40px', 'lineHeight': '14px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap', 'verticalAlign': 'top'},
                    style_data_conditional=[{'if': {'column_id': col}, 'width': '200px', 'maxWidth': '200px'} for col in df_response.columns],
                    style_header_conditional=[{'if': {'column_id': col}, 'width': '200px', 'maxWidth': '200px'} for col in df_response.columns],
                    tooltip_data=tooltip_data, tooltip_duration=None, fill_width=False
                )
                
                export_button = html.Button("Export as CSV", id={"type": "export-button", "index": table_uuid}, className="insight-button", style={'marginRight': '16px'})
                insight_button = html.Button("Analyze with AI", id={"type": "insight-button", "index": table_uuid}, className="insight-button")
                
                content_elements = []
                if description:
                    content_elements.append(dcc.Markdown(description, style={'marginBottom': '15px'}))
                content_elements.append(html.Div(data_table, style={'marginBottom': '10px'}))
                content_elements.append(html.Div([export_button, insight_button], style={'display': 'flex'}))
                content = html.Div(content_elements)
        
        # Case 3: The response is something else (e.g., empty DataFrame), so show no results.
        else:
            content = dcc.Markdown("Your request returned no results. This may happen if the data doesn’t exist or if you don’t have permission to view it.", className="message-text-bot")

        bot_response = html.Div([
            html.Div(html.Div(className="model-avatar"), className="model-info"),
            html.Div(content, className="message-content")
        ], className="bot-message message")
        
        updated_messages = current_messages[:-1] + [bot_response]
        if chat_history and session_data and session_data.get("current_session") is not None:
            current_session_index = session_data["current_session"]
            chat_history_list = list(chat_history) if isinstance(chat_history, tuple) else chat_history
            if 0 <= current_session_index < len(chat_history_list):
                chat_history_list[current_session_index]["messages"] = updated_messages
                if new_conv_id:
                    chat_history_list[current_session_index]["conversation_id"] = new_conv_id
                chat_history = chat_history_list
            
        return updated_messages, chat_history, {"trigger": False, "message": ""}, False, new_conv_id

    except DatabricksError as dbe:
        logger.error(f"Databricks API Error in get_model_response: {dbe}")
        error_msg = f"A service error occurred. Please try again later. (Details: {dbe.message})"
        error_response = html.Div([
            html.Div(html.Div(className="model-avatar"), className="model-info"),
            html.Div(html.Div(error_msg, className="message-text-bot"), className="message-content")
        ], className="bot-message message")
        
        updated_messages = current_messages[:-1] + [error_response]
        if chat_history and session_data and session_data.get("current_session") is not None:
            current_session_index = session_data["current_session"]
            chat_history_list = list(chat_history) if isinstance(chat_history, tuple) else chat_history
            if 0 <= current_session_index < len(chat_history_list):
                chat_history_list[current_session_index]["messages"] = updated_messages
                if new_conv_id:
                    chat_history_list[current_session_index]["conversation_id"] = new_conv_id
                chat_history = chat_history_list

        return updated_messages, chat_history, {"trigger": False, "message": ""}, False, new_conv_id

    except Exception as e:
        logger.error(f"Error in get_model_response: {e}")
        error_msg = f"Sorry, I encountered an error: {e}. Please try again later."
        error_response = html.Div([
            html.Div(html.Div(className="model-avatar"), className="model-info"),
            html.Div(html.Div(error_msg, className="message-text-bot"), className="message-content")
        ], className="bot-message message")
        
        updated_messages = current_messages[:-1] + [error_response]
        if chat_history and session_data and session_data.get("current_session") is not None:
            current_session_index = session_data["current_session"]
            chat_history_list = list(chat_history) if isinstance(chat_history, tuple) else chat_history
            if 0 <= current_session_index < len(chat_history_list):
                chat_history_list[current_session_index]["messages"] = updated_messages
                if new_conv_id:
                    chat_history_list[current_session_index]["conversation_id"] = new_conv_id
                chat_history = chat_history_list

        return updated_messages, chat_history, {"trigger": False, "message": ""}, False, new_conv_id

###
# CALLBACK: Export DataFrame to CSV
#
# This callback handles the "Export as CSV" button click. It retrieves the
# corresponding DataFrame from the server-side cache using its UUID and
# sends it to the user's browser for download.
###
@app.callback(
    Output("download-dataframe-csv", "data"),
    Output("processed-export-clicks", "data"),
    Input({"type": "export-button", "index": ALL}, "n_clicks"),
    State("processed-export-clicks", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks_list, processed_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, no_update

    triggered_input = ctx.triggered[0]
    button_id_dict = json.loads(triggered_input["prop_id"].split(".")[0])
    table_uuid = button_id_dict["index"]
    n_clicks = triggered_input["value"]

    if not n_clicks or processed_clicks.get(table_uuid) == n_clicks:
        return dash.no_update, no_update

    # Retrieve the DataFrame as a CSV string from the diskcache
    df_csv_string = df_cache_for_long_callbacks.get(table_uuid)
    if df_csv_string is None:
        return dash.no_update, no_update

    # Reconstruct the DataFrame from the CSV string
    df = pd.read_csv(StringIO(df_csv_string))
    
    processed_clicks[table_uuid] = n_clicks
    return dcc.send_data_frame(df.to_csv, f"exported_data_{table_uuid[:8]}.csv", index=False), processed_clicks

###
# CALLBACK: Toggle Sidebar Visibility
#
# Manages the opening and closing of the sidebar. It adjusts the CSS classes
# of several components to create a smooth sliding animation and shift the
# main content accordingly.
###
@app.callback(
    [Output("sidebar", "className"),
     Output("new-chat-button", "style"),
     Output("sidebar-new-chat-button", "style"),
     Output("change-space-button", "style"),
     Output("logo-container", "className"),
     Output("nav-left", "className"),
     Output("left-component", "className"),
     Output("main-content", "className")],
    Input("sidebar-toggle", "n_clicks"),
    [State("sidebar", "className"),
     State("left-component", "className"),
     State("main-content", "className")]
)
def toggle_sidebar(n_clicks, current_sidebar_class, current_left_component_class, current_main_content_class):
    if n_clicks:
        if "sidebar-open" in current_sidebar_class:
            return "sidebar", {"display": "flex"}, {"display": "none"}, {"display": "flex"}, "logo-container", "nav-left", "left-component", "main-content"
        else:
            return "sidebar sidebar-open", {"display": "none"}, {"display": "flex"}, {"display": "none"}, "logo-container logo-container-open", "nav-left nav-left-open", "left-component left-component-open", "main-content main-content-shifted"
    return current_sidebar_class, {"display": "flex"}, {"display": "none"}, {"display": "flex"}, "logo-container", "nav-left", "left-component", current_main_content_class

###
# CALLBACK: Display Selected Chat History
#
# When a user clicks on a past conversation in the sidebar's chat list, this
# callback retrieves the corresponding message history from the `chat-history-store`
# and displays it in the main chat window.
###
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("conversation-id-store", "data", allow_duplicate=True)],
    Input({"type": "chat-item", "index": ALL}, "n_clicks"),
    [State("chat-history-store", "data"),
     State("chat-list", "children"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def show_chat_history(n_clicks, chat_history, current_chat_list, session_data):
    ctx = dash.callback_context
    if not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    clicked_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
    clicked_index = clicked_id["index"]
    
    if not chat_history or clicked_index >= len(chat_history):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    new_session_data = {"current_session": clicked_index}
    
    conversation_id_to_restore = chat_history[clicked_index].get("conversation_id")

    updated_chat_list = []
    for i, item in enumerate(current_chat_list):
        item['props']['className'] = "chat-item active" if i == clicked_index else "chat-item"
        updated_chat_list.append(item)

    return (chat_history[clicked_index]["messages"], "welcome-container hidden",
            updated_chat_list, new_session_data, conversation_id_to_restore)

###
# CALLBACK: Restore Session on Page Load
#
# When the app is refreshed, this callback restores the chat UI from the
# state persisted in session storage. It populates the chat messages and
# the sidebar chat list, ensuring a seamless user experience.
###
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True)],
    Input("main-content", "style"), # Trigger after main chat UI becomes visible
    [State("chat-history-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def restore_session_on_load(main_style, chat_history, session_data):
    # This callback runs when the main content area is shown (e.g., after agent selection or on page reload with a selected agent)
    if not main_style or main_style.get("display") == "none":
        return dash.no_update, dash.no_update, dash.no_update

    # Check if there's any history or an active session to restore
    if not chat_history or not session_data or session_data.get("current_session") is None:
        return dash.no_update, dash.no_update, dash.no_update

    current_session_index = session_data.get("current_session")

    # Validate the session index against the chat history length
    if not isinstance(current_session_index, int) or current_session_index >= len(chat_history):
        return dash.no_update, dash.no_update, dash.no_update

    # Restore the chat messages for the active session
    messages = chat_history[current_session_index].get("messages", [])

    # Rebuild the list of conversations in the sidebar
    chat_list = [
        html.Div(
            session["queries"][0],
            className=f"chat-item{' active' if i == current_session_index else ''}",
            id={"type": "chat-item", "index": i}
        )
        for i, session in enumerate(chat_history)
    ]
    
    # Hide the welcome container if there are messages to display
    welcome_class = "welcome-container hidden" if messages else "welcome-container visible"

    return messages, chat_list, welcome_class

###
# CALLBACK: Auto-scroll Chat Window
#
# This is a clientside callback that runs in the user's browser. It automatically
# scrolls the chat message container to the bottom whenever new messages are added,
# ensuring the latest message is always visible.
###
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }, 100);
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    Input('chat-messages', 'children'),
    prevent_initial_call=True
)

###
# CALLBACK: Start a New Chat
#
# Resets the chat interface to its initial state when the "New Chat" button is
# clicked. It clears the message display and resets the session state, allowing
# the user to start a fresh conversation.
###
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("conversation-id-store", "data", allow_duplicate=True)],
    [Input("new-chat-button", "n_clicks"),
     Input("sidebar-new-chat-button", "n_clicks")],
    State("chat-history-store", "data"),
    prevent_initial_call=True
)
def reset_to_welcome(n_clicks1, n_clicks2, chat_history_store):
    if not callback_context.triggered:
        return [no_update] * 6
    
    return [], {"trigger": False, "message": ""}, False, chat_history_store, {"current_session": None}, None

###
# CALLBACK: Change Agent and Reset UI
#
# When the "Change Agent" button is clicked, this callback resets the application
# state to show the agent selection overlay, allowing the user to choose a
# different BI agent.
###
@app.callback(
    [Output("selected-space-id", "data", allow_duplicate=True),
     Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("conversation-id-store", "data", allow_duplicate=True)],
    Input("change-space-button", "n_clicks"),
    State("chat-history-store", "data"),
    prevent_initial_call=True
)
def change_space_and_reset(n_clicks, chat_history):
    if not n_clicks:
        return [no_update] * 7
    return None, [], {"trigger": False, "message": ""}, False, chat_history, {"current_session": None}, None

###
# CALLBACK: Show/Hide Welcome Container
#
# This callback controls the visibility of the initial welcome screen. It hides
# the welcome message container as soon as the chat history is populated with
# any messages.
###
@app.callback(
    Output("welcome-container", "className", allow_duplicate=True),
    Input("chat-messages", "children"),
    prevent_initial_call=True
)
def reset_query_running(chat_messages):
    return "welcome-container hidden" if chat_messages else "welcome-container visible"

###
# CALLBACK: Disable Inputs While Query is Running
#
# To prevent multiple submissions, this callback disables the chat input field
# and send/new-chat buttons whenever a query is being processed by the backend.
# This callback will no longer be strictly necessary for get_model_response
# and confirm_and_generate_insights due to `running` arg in long_callback.
# However, it might still be useful for other parts of the app.
###
@app.callback(
    [Output("chat-input-fixed", "disabled"),
     Output("mic-button", "disabled"),
     Output("send-button-fixed", "disabled"),
     Output("new-chat-button", "disabled"),
     Output("sidebar-new-chat-button", "disabled")],
    Input("query-running-store", "data")
)
def toggle_input_disabled(query_running):
    # This callback now acts as a fallback/general input disabler.
    # The `running` argument in long_callback handles specific disabling
    # during its execution.
    return [query_running] * 5

###
# CALLBACK: Open Insight Modal
#
# When the "Analyze with AI" button is clicked, this callback opens the
# insight prompt modal and stores the UUID of the relevant DataFrame.
###
@app.callback(
    [Output("insight-prompt-modal", "is_open"),
     Output("current-dataframe-uuid", "data"),
     Output("processed-insight-clicks", "data")],
    Input({"type": "insight-button", "index": ALL}, "n_clicks"),
    State("processed-insight-clicks", "data"),
    prevent_initial_call=True
)
def open_insight_modal(n_clicks_list, processed_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    triggered_input = ctx.triggered[0]
    button_id_dict = json.loads(triggered_input["prop_id"].split(".")[0])
    table_uuid = button_id_dict["index"]
    n_clicks = triggered_input["value"]

    if not n_clicks or processed_clicks.get(table_uuid) == n_clicks:
        return no_update, no_update, no_update

    processed_clicks[table_uuid] = n_clicks
    return True, table_uuid, processed_clicks

###
# CALLBACK: Control Insight Modal Confirm Button
#
# Disables the "Confirm" button in the insight modal if the prompt
# textarea is empty, ensuring a prompt is always provided.
###
@app.callback(
    Output("confirm-insight-prompt-button", "disabled"),
    Input("insight-prompt-textarea", "value")
)
def set_confirm_button_disabled(textarea_value):
    return not bool(textarea_value and textarea_value.strip())

###
# CALLBACK: Close Insight Modal on Confirm
#
# Provides a better user experience by immediately closing the insight modal
# when the "Confirm" button is clicked, allowing the insight generation to
# proceed in the background.
###
@app.callback(
    Output("insight-prompt-modal", "is_open", allow_duplicate=True),
    Input("confirm-insight-prompt-button", "n_clicks"),
    prevent_initial_call=True
)
def close_modal_on_confirm(n_clicks):
    return False if n_clicks else no_update

###
# CALLBACK: Trigger Insight Generation
#
# This is the first step in the insight generation chain. After the user
# confirms the prompt, this callback adds a "Generating insights..." indicator
# to the chat and triggers the final processing callback.
###
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("insight-trigger-store", "data")],
    Input("confirm-insight-prompt-button", "n_clicks"),
    [State("insight-prompt-textarea", "value"),
     State("current-dataframe-uuid", "data"),
     State("chat-messages", "children"),
     State("query-running-store", "data")],
    prevent_initial_call=True
)
def trigger_insight_generation(n_clicks, prompt_value, table_uuid, current_messages, query_running):
    if not n_clicks or not table_uuid or query_running:
        return no_update, no_update, no_update

    thinking_indicator = html.Div(
        html.Div([html.Span(className="spinner"), html.Span("Generating insights...")], className="thinking-indicator"),
        className="bot-message message"
    )
    updated_messages = (current_messages or []) + [thinking_indicator]
    trigger_data = {"table_uuid": table_uuid, "prompt_value": prompt_value}
    
    return updated_messages, True, trigger_data

###
# CALLBACK: Generate and Display Insights
#
# This is the final step in the insight generation chain. It retrieves the
# DataFrame from the server-side cache, calls the LLM to generate insights,
# and then replaces the "Generating..." indicator with the final result.
###
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("insight-trigger-store", "data", allow_duplicate=True)],
    Input("insight-trigger-store", "data"),
    [State("chat-history-store", "data"),
     State("chat-messages", "children"),
     State("session-store", "data")],
    prevent_initial_call=True,
    background=True,
    # Define outputs to be updated while the callback is running
    running=[
        (Output("chat-input-fixed", "disabled"), True, False),
        (Output("mic-button", "disabled"), True, False),
        (Output("send-button-fixed", "disabled"), True, False),
        (Output("new-chat-button", "disabled"), True, False),
        (Output("sidebar-new-chat-button", "disabled"), True, False),
    ],
)
def confirm_and_generate_insights(trigger_data, chat_history, current_messages, session_data):
    # This callback can now be long-running without blocking the main Dash thread.
    if not trigger_data:
        return no_update, no_update, no_update, no_update

    table_uuid = trigger_data["table_uuid"]
    prompt_value = trigger_data["prompt_value"]
    messages_without_thinking = current_messages[:-1]

    # Retrieve the DataFrame as a CSV string from the diskcache
    df_csv_string = df_cache_for_long_callbacks.get(table_uuid)
    if df_csv_string is None:
        error_msg = "Error: Data not found for insights."
        error_response = html.Div(html.Div(html.Div(error_msg, className="message-text-bot"), className="message-content"), className="bot-message message")
        updated_messages = messages_without_thinking + [error_response]
    else:
        try:
            # Pass the DataFrame as a CSV string to the cached function (df_csv_string is already CSV)
            insights = call_llm_for_insights(df_csv_string, prompt=prompt_value)
            # Create the "Ask follow-up" button
            ask_follow_up_button = html.Button(
                "Ask follow-up", 
                id={"type": "insight-button", "index": table_uuid},
                className="insight-button", 
                style={'marginTop': '10px'}
            )
            insight_message = html.Div([
                html.Div(html.Div(className="model-avatar"), className="model-info"),
                html.Div([
                    dcc.Markdown(insights),
                    ask_follow_up_button
                ], className="message-content")
            ], className="bot-message message", id=f"insight-response-{uuid.uuid4()}")
            updated_messages = messages_without_thinking + [insight_message]
        except Exception as e:
            error_msg = f"Error generating insights: {str(e)}"
            error_response = html.Div(html.Div(html.Div(error_msg, className="message-text-bot"), className="message-content"), className="bot-message message")
            updated_messages = messages_without_thinking + [error_response]
    
    if chat_history and session_data and session_data.get("current_session") is not None:
        current_session_index = session_data["current_session"]
        # Ensure chat_history is a mutable list. If it's a tuple from the state, convert it.
        chat_history_list = list(chat_history) if isinstance(chat_history, tuple) else chat_history
        if 0 <= current_session_index < len(chat_history_list):
            chat_history_list[current_session_index]["messages"] = updated_messages
        chat_history = chat_history_list
        
    return updated_messages, False, chat_history, None

###
# CALLBACK: Fetch Available Agents (Spaces)
#
# On application load, this callback contacts the Genie API to get a list of
# all available BI agents (called "spaces") that the user can interact with.
###
@app.callback(
    Output("spaces-list", "data"),
    Input("space-select-container", "id")
)
def fetch_spaces(_):
    try:
        headers = request.headers
        token = headers.get('X-Forwarded-Access-Token')
        host = os.environ.get("DATABRICKS_HOST")
        client = GenieClient(host=host, space_id="", token=token)
        return client.list_spaces()
    except Exception as e:
        logger.error(f"Failed to fetch spaces: {e}")
        return []

###
# CALLBACK: Populate Agent Selection Dropdown
#
# Populates the dropdown on the initial selection screen with the list of
# agents fetched by the `fetch_spaces` callback.
###
@app.callback(
    Output("space-dropdown", "options"),
    Input("spaces-list", "data")
)
def update_space_dropdown(spaces):
    if not spaces:
        return [{"label": "No available agents", "value": "no_spaces_found", "disabled": True}]
    return [{"label": s.get('title', ''), "value": s.get('space_id', '')} for s in spaces]

###
# CALLBACK: Enable Agent Selection Button
#
# Enables the "Explore Agent" button only after an agent has been
# selected from the dropdown menu.
###
@app.callback(
    Output("select-space-button", "disabled"),
    Input("space-dropdown", "value")
)
def enable_select_space_button(selected_value):
    return selected_value is None

###
# CALLBACK: Handle Agent Selection
#
# Once the user selects an agent and clicks "Explore Agent", this callback
# stores the chosen agent's ID, hides the selection overlay, and displays
# the main chat interface.
###
@app.callback(
    [Output("selected-space-id", "data", allow_duplicate=True),
     Output("space-select-container", "style"),
     Output("main-content", "style"),
     Output("welcome-title", "children", allow_duplicate=True),
     Output("welcome-description", "children", allow_duplicate=True)],
    Input("select-space-button", "n_clicks"),
    [State("space-dropdown", "value"),
     State("spaces-list", "data")],
    prevent_initial_call=True
)
def select_space(n_clicks, space_id, spaces):
    if not n_clicks or not space_id:
        return no_update, no_update, no_update, no_update, no_update
        
    selected = next((s for s in spaces if s["space_id"] == space_id), {})
    title = selected.get("title")
    description = selected.get("description")
    return space_id, {"display": "none"}, {"display": "block"}, title, description

###
# CALLBACK: Toggle Main UI Visibility
#
# Controls which main view is active: the agent selection overlay or the
# main chat interface with its top navigation bar.
###
@app.callback(
    [Output("main-content", "style", allow_duplicate=True),
     Output("space-select-container", "style", allow_duplicate=True),
     Output("top-nav", "style")],
    Input("selected-space-id", "data"),
    prevent_initial_call=True
)
def toggle_main_ui(selected_space_id):
    if selected_space_id:
        main_style = {"display": "block"}
        overlay_style = {"display": "none"}
        nav_top_style = {"display": "flex", "position": "fixed", "top": "0", "left": "0", "width": "100%", "zIndex": "1001"}
    else:
        main_style = {"display": "none"}
        overlay_style = {"display": "flex"}
        nav_top_style = {"display": "none"}
    return main_style, overlay_style, nav_top_style

###
# CALLBACK: Auto-scroll on Insight Generation
#
# A clientside callback to ensure the chat window scrolls to the bottom
# after a potentially long insight response is generated and displayed.
###
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var chatMessages = document.getElementById('chat-messages');
            if (chatMessages && chatMessages.lastElementChild) {
                chatMessages.lastElementChild.scrollIntoView({behavior: 'auto', block: 'end'});
            }
        }, 100);
        return '';
    }
    """,
    Output('dummy-insight-scroll', 'children'),
    Input({'type': 'insight-response', 'index': ALL}, 'children'),
    prevent_initial_call=True
)

###
# CALLBACK: Manage Query Submission Tooltip
#
# Shows or hides a tooltip over the chat input to inform the user that
# they cannot submit a new query while one is already running.
###
@app.callback(
    Output("query-tooltip", "className"),
    Input("query-running-store", "data")
)
def update_query_tooltip_class(query_running):
    return "query-tooltip query-tooltip-active" if query_running else "query-tooltip"

###
# CALLBACK: Fetch and Store User Information
#
# On initial app load, this callback retrieves the logged-in user's information
# (name, email) from the request headers provided by the authentication proxy.
###
@app.callback(
    Output("username-store", "data"),
    Input("root-container", "children")
)
def fetch_username(_):
    try:
        email = request.headers.get("X-Forwarded-Preferred-Username", "")
        if not email:
            return {'display_name': 'User', 'email': '', 'initial': 'U'}

        username_part = email.split("@")[0]
        name_parts = username_part.split(".")
        display_name = " ".join([part.capitalize() for part in name_parts])
        initial = display_name[0] if display_name else 'U'
        
        return {'display_name': display_name, 'email': email, 'initial': initial}
    except Exception as e:
        logger.error(f"Error fetching username: {e}")
        return {'display_name': 'User', 'email': '', 'initial': 'U'}

###
# CALLBACK: Fetch and Store User Token for Long Callbacks
# This new callback fetches the X-Forwarded-Access-Token and stores it
# in a dcc.Store so it can be passed to long callbacks, which don't have
# direct access to flask.request.headers.
###
@app.callback(
    Output("user-token-store", "data"),
    Input("root-container", "children"), # Trigger on app load
    prevent_initial_call=False # Allow initial call to populate
)
def get_user_token_on_load(_):
    try:
        token = request.headers.get("X-Forwarded-Access-Token")
        logger.info(f"Fetched X-Forwarded-Access-Token: {token[:10]}...") # Log first 10 chars for debug
        return token
    except Exception as e:
        logger.error(f"Error fetching X-Forwarded-Access-Token: {e}")
        return None

###
# CALLBACK: Update UI with User Information
#
# Populates various parts of the UI with the fetched user's name, email,
# and avatar initials, creating a personalized experience.
###
@app.callback(
    [Output("welcome-user-greeting", "children"),
     Output("user-avatar-initials", "children"),
     Output("user-email-display", "children"),
     Output("user-avatar-initials-overlay", "children"),
     Output("user-email-display-overlay", "children")],
    Input("username-store", "data")
)
def update_username_display(user_data):
    if not user_data:
        return None, 'U', '', 'U', ''
    
    display_name = user_data.get('display_name', 'User')
    email = user_data.get('email', '')
    initial = user_data.get('initial', 'U')
    first_name = display_name.split(" ")[0]
    greeting = f"Hello, {first_name}"
    
    return greeting, initial, email, initial, email

###
# SCRIPT EXECUTION
#
# This standard Python construct ensures that the Dash development server
# is started only when the script is executed directly.
###
if __name__ == "__main__":
    app.run(debug=False)