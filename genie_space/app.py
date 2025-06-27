import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context, no_update, clientside_callback, dash_table
import dash_bootstrap_components as dbc
import json
from genie_room import genie_query
import pandas as pd
import os
from dotenv import load_dotenv
import sqlparse
from flask import request
import logging
from genie_room import GenieClient
import os
import uuid
from io import StringIO
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.config import Config
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="BI Agent"
)

# Define the layout
app.layout = html.Div([
    html.Div([
        dcc.Store(id="selected-space-id", data=None),
        dcc.Store(id="spaces-list", data=[]),
        dcc.Store(id="conversation-id-store", data=None),
        dcc.Store(id="username-store", data=None),
        dcc.Store(id="current-dataframe-uuid", data=None), # New Store to hold the UUID of the DataFrame for insight generation
        dcc.Store(id="processed-export-clicks", data={}), # To store uuids of exported tables
        dcc.Store(id="processed-insight-clicks", data={}), # To store uuids of insight generated tables
        dcc.Store(id="processed-confirm-insight-clicks", data={}), # New Store for processed confirm insight clicks
        # Top navigation bar - now fixed at the top
        html.Div([
            # Left component containing both nav-left and sidebar
            html.Div([
                # Nav left
                html.Div([
                    html.Button([
                        html.Img(src="assets/menu_icon.svg", className="menu-icon")
                    ], id="sidebar-toggle", className="nav-button"),
                    html.Button([
                        html.Img(src="assets/plus_icon.svg", className="new-chat-icon")
                    ], id="new-chat-button", className="nav-button",disabled=False, title="New chat"),
                    html.Button([
                        html.Img(src="assets/plus_icon.svg", className="new-chat-icon"),
                        html.Div("New chat", className="new-chat-text")
                    ], id="sidebar-new-chat-button", className="new-chat-button",disabled=False),
                    html.Button([
                        html.Img(src="assets/change.png", style={'height': '16px'})
                    ], id="change-space-button", className="nav-button",disabled=False, title="Change Agent")

                ], id="nav-left", className="nav-left"),

                # Sidebar
                html.Div([
                    html.Div([
                        html.Div("Your recent conversations", className="sidebar-header-text"),
                    ], className="sidebar-header"),
                    html.Div([], className="chat-list", id="chat-list")
                ], id="sidebar", className="sidebar")
            ], id="left-component", className="left-component"),

            html.Div([
                html.Div([
                    html.Div(className="company-logo-black")
                    ], id="logo-container",
                    className="logo-container"
                )
            ], id="nav-center", className="nav-center"),
            html.Div([
                html.Div(className="user-avatar"),
                html.Div(
                    id="username-display-nav",
                    className="username-display",
                    style={'color': 'black'}
                ),
                html.A(
                    html.Button([html.Img(src="assets/logout_icon_black.svg")],
                        id="logout-button-nav",
                        className="logout-button-black",
                        title="Logout"
                    ),
                    href=f"https://{os.getenv('DATABRICKS_HOST')}/login.html",
                    className="logout-link"
                )
            ], id="nav-right", className="nav-right"),
        ], id="top-nav", className="top-nav", style={"display": "none"}), # Initially hidden

        # Main content wrapper (includes space overlay and main chat)
        # This wrapper will have a margin-top equal to the fixed header's height
        html.Div([
            # Space selection overlay - now positioned relative to this new wrapper's top
            html.Div([
                html.Div([
                    html.Div(className="company-logo"),
                    html.Div([
                        html.Div(className="user-avatar"),
                        html.Div(
                            id="username-display-overlay",
                            className="username-display",
                            style={'color': 'white'}
                        ),
                        html.A(
                            html.Button([html.Img(src="assets/logout_icon.svg")],
                                id="logout-button-overlay",
                                className="logout-button",
                                title="Logout"
                            ),
                            href=f"https://{os.getenv('DATABRICKS_HOST')}/login.html",
                            className="logout-link"
                        )
                    ], className="nav-right"),
                    html.Div("BI Agent Platform", className="main-title"),
                    html.Div("Empowering insights through conversation", className="space-select-tagline"),
                    html.Div(id="welcome-user-greeting", className="greeting-title"),
                    html.Div([
                        dcc.Dropdown(id="space-dropdown", options=[], placeholder="Choose an Agent", className="space-select-dropdown", optionHeight=40, searchable=True,
                        style={'z-index': 1000}
                        ),
                        html.Button("Explore Agent", id="select-space-button", className="space-select-button", disabled= True)
                        ],className="explore-agent-container"),
                    html.Div([
                        html.A(
                            html.Button("Documentation", className="request-support-button"),
                            href="https://jda365.sharepoint.com/:b:/r/sites/O365-AnalyticsJDA/Shared%20Documents/Cognitive%20Analytics/BI%20Agent%20Documentation.pdf?csf=1&web=1&e=7xdXIe",
                            target="_blank"
                        ),
                        html.Span("|", style={'color': 'white', 'fontSize': '1em', 'padding': '0 0.5em'}),
                        html.A(
                            html.Button("Request Support", className="request-support-button"),
                            href="https://jdaswin.service-now.com/jdasp/?id=sc_cat_item&sys_id=997bc9dc1baea590766441dce54bcb6e&sysparm_category=afc4a6211b6aa1d0766441dce54bcbd6",
                            target="_blank"
                        )
                    ], className="support-links"),
                    html.Div(id="space-select-error", className="space-select-error")
                ], className="space-select-card")
            ], id="space-select-container", className="space-select-container", style={"height": "100%", "top": "0"}),

            # Main content area
            html.Div([
                html.Div([
                    # Chat content
                    html.Div([
                        # Welcome container
                        html.Div([
                            html.Div([html.Div([
                            html.Div(className="genie-logo")
                        ], className="genie-logo-container")],
                        className="genie-logo-container-header"),

                            # Add settings button with tooltip
                            html.Div([
                                html.Div(id="welcome-title", className="welcome-message"),
                            ], className="welcome-title-container"),

                            html.Div(id="welcome-description",
                                    className="welcome-message-description"),

                            # Suggestion buttons with IDs
                            html.Div([
                                html.Button([
                                    html.Div(className="suggestion-icon"),
                                    html.Div("What is the purpose of this Agent? Give me a short summary.",
                                           className="suggestion-text", id="suggestion-1-text")
                                ], id="suggestion-1", className="suggestion-button"),
                                html.Button([
                                    html.Div(className="suggestion-icon"),
                                    html.Div("How to converse with the Agent? Give me an example prompt.",
                                           className="suggestion-text", id="suggestion-2-text")
                                ], id="suggestion-2", className="suggestion-button"),
                                html.Button([
                                    html.Div(className="suggestion-icon"),
                                    html.Div("Explain the dataset behind this Agent.",
                                           className="suggestion-text", id="suggestion-3-text")
                                ], id="suggestion-3", className="suggestion-button"),
                                html.Button([
                                    html.Div(className="suggestion-icon"),
                                    html.Div("What columns or fields are available in this dataset?",
                                           className="suggestion-text", id="suggestion-4-text")
                                ], id="suggestion-4", className="suggestion-button")
                            ], className="suggestion-buttons")
                        ], id="welcome-container", className="welcome-container visible"),

                        # Chat messages
                        html.Div([], id="chat-messages", className="chat-messages"),
                    ], id="chat-content", className="chat-content"),

                    # Input area
                    html.Div([
                        html.Div([
                            dcc.Input(
                                id="chat-input-fixed",
                                placeholder="Ask your question...",
                                className="chat-input",
                                type="text",
                                disabled=False
                            ),
                            html.Div([
                                # ADDED: Microphone button for voice dictation
                                html.Button(
                                    html.Img(src="assets/mic_icon.svg", style={'width': '18px', 'height': '18px'}),
                                    id="mic-button",
                                    className="input-button",
                                    title="Start/Stop Dictation"
                                ),
                                html.Button(
                                    id="send-button-fixed",
                                    className="input-button send-button",
                                    disabled=False
                                )
                            ], className="input-buttons-right"),
                            html.Div("You can only submit one query at a time",
                                    id="query-tooltip",
                                    className="query-tooltip")
                        ], id="fixed-input-container", className="fixed-input-container"),
                        html.Div("Always review the accuracy of responses.", className="disclaimer-fixed")
                    ], id="fixed-input-wrapper", className="fixed-input-wrapper"),
                ], id="chat-container", className="chat-container"),
            ], id="main-content", className="main-content", style={"display": "none"}), # display is controlled by callbacks
        ], style={"height": "100vh", "position": "relative"}), # New wrapper for main content and overlay

        html.Div(id='dummy-output'),
        dcc.Store(id="chat-trigger", data={"trigger": False, "message": ""}),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Store(id="query-running-store", data=False),
        dcc.Store(id="session-store", data={"current_session": None}),
        html.Div(id='dummy-insight-scroll'),
        dcc.Download(id="download-dataframe-csv"),

        # New Insight Prompt Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Generate Insights")),
                dbc.ModalBody(
                    [
                        html.P("Review and modify the prompt for generating insights:"),
                        dcc.Textarea(
                            id="insight-prompt-textarea",
                            value=(
                                "You are a professional data analyst. Given the following table data, provide deep, actionable analysis for\n"
                                "1. Key insights and trends.\n"
                                "2. Notable patterns\n"
                                "3. Business implications.\n"
                                "Be thorough, professional, and concise.\n\n"
                            ),
                            style={"width": "100%", "height": "200px"},
                        ),
                        html.Div(id="insight-generation-status"), # To show loading/error messages
                    ]
                ),
                dbc.ModalFooter([dbc.Button("Confirm", id="confirm-insight-prompt-button", className="confirm-button", n_clicks=0)],
                style={
                    "backgroundColor": "#e6e6e62e",
                    "padding": "10px",
                    "borderTop": "1px solid #ccc"
                }
                ),
            ],
            id="insight-prompt-modal",
            is_open=False,
            size="lg",
            centered=True,
        ),
    ], id="app-inner-layout"),
], id="root-container")

# Store chat history
chat_history = []

def format_sql_query(sql_query):
    """Format SQL query using sqlparse library"""
    formatted_sql = sqlparse.format(
        sql_query,
        keyword_case='upper',  # Makes keywords uppercase
        identifier_case=None,  # Preserves identifier case
        reindent=True,         # Adds proper indentation
        indent_width=2,        # Indentation width
        strip_comments=False,  # Preserves comments
        comma_first=False      # Commas at the end of line, not beginning
    )
    return formatted_sql

def call_llm_for_insights(df, prompt=None):
    """
    Call an LLM to generate insights from a DataFrame.
    Args:
        df: pandas DataFrame
        prompt: Optional custom prompt
    Returns:
        str: Insights generated by the LLM
    """
    if prompt is None:
        prompt = (
            "You are a professional data analyst. Given the following table data, provide deep, actionable analysis for"
            "1. Key insights and trends."
            "2. Notable patterns"
            "3. Business implications."
            "Be thorough, professional, and concise.\n\n"
        )
    csv_data = df.to_csv(index=False)
    full_prompt = f"{prompt}Table data:\n{csv_data}"
    # Call OpenAI (replace with your own LLM provider as needed)
    try:
        client = WorkspaceClient()
        response = client.serving_endpoints.query(
            os.getenv("SERVING_ENDPOINT_NAME"),
            messages=[ChatMessage(content=full_prompt, role=ChatMessageRole.USER)],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# First callback: Handle inputs and show thinking indicator
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

    # Handle suggestion buttons
    suggestion_map = {
        "suggestion-1": s1_text,
        "suggestion-2": s2_text,
        "suggestion-3": s3_text,
        "suggestion-4": s4_text
    }

    # Get the user input based on what triggered the callback
    if trigger_id in suggestion_map:
        user_input = suggestion_map[trigger_id]
    else:
        user_input = input_value

    if not user_input:
        return [no_update] * 8

    # Create user message with user info
    user_message = html.Div([
        html.Div(user_input, className="message-text"),
        html.Div([
            html.Div(className="user-avatar")
        ], className="user-info")
    ], className="user-message message")

    # Add the user message to the chat
    updated_messages = current_messages + [user_message] if current_messages else [user_message]

    # Add thinking indicator
    thinking_indicator = html.Div([
        html.Div([
            html.Span(className="spinner"),
            html.Span("Thinking...")
        ], className="thinking-indicator")
    ], className="bot-message message")

    updated_messages.append(thinking_indicator)

    # Handle session management
    if session_data["current_session"] is None:
        session_data = {"current_session": len(chat_history) if chat_history else 0}

    current_session = session_data["current_session"]

    # Update chat history
    if chat_history is None:
        chat_history = []

    if current_session < len(chat_history):
        chat_history[current_session]["messages"] = updated_messages
        chat_history[current_session]["queries"].append(user_input)
    else:
        chat_history.insert(0, {
            "session_id": current_session,
            "queries": [user_input],
            "messages": updated_messages
        })

    # Update chat list
    updated_chat_list = []
    for i, session in enumerate(chat_history):
        first_query = session["queries"][0]
        is_active = (i == current_session)
        updated_chat_list.append(
            html.Div(
                first_query,
                className=f"chat-item{'active' if is_active else ''}",
                id={"type": "chat-item", "index": i}
            )
        )

    return (updated_messages, "", "welcome-container hidden",
            {"trigger": True, "message": user_input}, True,
            updated_chat_list, chat_history, session_data)

# Second callback: Make API call and show response
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("conversation-id-store", "data", allow_duplicate=True)],
    [Input("chat-trigger", "data")],
    [State("chat-messages", "children"),
     State("chat-history-store", "data"),
     State("selected-space-id", "data"),
     State("conversation-id-store", "data")],
    prevent_initial_call=True
)
def get_model_response(trigger_data, current_messages, chat_history, selected_space_id, conversation_id):
    if not trigger_data or not trigger_data.get("trigger"):
        return no_update, no_update, no_update, no_update, no_update

    user_input = trigger_data.get("message", "")
    if not user_input:
        return no_update, no_update, no_update, no_update, no_update

    new_conv_id = conversation_id
    try:
        headers = request.headers
        user_token = headers.get('X-Forwarded-Access-Token')
        new_conv_id, response, query_text = genie_query(user_input, user_token, selected_space_id, conversation_id)

        # Store the DataFrame in chat_history for later retrieval by insight button
        df = pd.DataFrame(response) if not isinstance(response, str) else None
        if df is not None:
            if chat_history and len(chat_history) > 0:
                table_uuid = str(uuid.uuid4())
                chat_history[0].setdefault('dataframes', {})[table_uuid] = df.to_json(orient='split')
            else:
                # If chat_history is empty, initialize it with the new dataframe
                table_uuid = str(uuid.uuid4())
                chat_history = [{"dataframes": {table_uuid: df.to_json(orient='split')}, "messages": [], "queries": []}]
        else:
            table_uuid = None # No df to store

        if isinstance(response, str):
            if response == user_input:
                content = dcc.Markdown("Your request returned no results. This may happen if the data doesn’t exist or if you don’t have permission to view it.", className="message-text-bot")
            else:
                # Escape square brackets to prevent markdown auto-linking
                import re
                processed_response = response
            
                # Escape all square brackets to prevent markdown from interpreting them as links
                processed_response = processed_response.replace('[', '\\[').replace(']', '\\]')
            
                # Escape parentheses to prevent markdown from interpreting them as links
                processed_response = processed_response.replace('(', '\\(').replace(')', '\\)')
            
                # Escape angle brackets to prevent markdown from interpreting them as links
                processed_response = processed_response.replace('<', '\\<').replace('>', '\\>')
            
                content = dcc.Markdown(processed_response, className="message-text-bot")
        else:
            df_response = pd.DataFrame(response)
            if df_response.shape == (1, 1):
                markdown_response = str(df_response.iloc[0, 0])
                query_section = None
                if query_text:
                    formatted_sql = format_sql_query(query_text)
                    query_index = f"{len(chat_history)}-{len(current_messages)}"
                    query_section = html.Div([
                        html.Div([
                            html.Button([
                                html.Span("Show code", id={"type": "toggle-text", "index": query_index}, style={"display": "none"})
                            ], id={"type": "toggle-query", "index": query_index}, className="toggle-query-button", n_clicks=0)
                        ], className="toggle-query-container"),
                        html.Div([
                            html.Pre([html.Code(formatted_sql, className="sql-code")], className="sql-pre")
                        ], id={"type": "query-code", "index": query_index}, className="query-code-container hidden")
                    ], id={"type": "query-section", "index": query_index}, className="query-section")

                content = html.Div([
                    dcc.Markdown(markdown_response)
                    # , query_section
                ]) if query_section else dcc.Markdown(markdown_response)

            else:
                table_data = df_response.to_dict('records')
                table_columns = [{"name": col, "id": col} for col in df_response.columns]

                tooltip_data = [
                    {col: {'value': str(row[col]), 'type': 'markdown'} for col in df_response.columns}
                    for row in table_data
                ]
                # header_tooltips = {col: {'value': col, 'type': 'markdown'} for col in df_response.columns}

                data_table = dash_table.DataTable(
                    id=f"table-{len(chat_history)}",
                    data=table_data,
                    columns=table_columns,
                    sort_action="native",
                    filter_action="native",
                    style_table={'maxHeight': '300px', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '95%'},
                    style_data={
                        'textAlign': 'left',
                        'padding': '5px', 'height': '40px', 'maxHeight': '40px',
                        'lineHeight': '14px', 'overflow': 'hidden', 'textOverflow': 'ellipsis',
                        'whiteSpace': 'nowrap', 'verticalAlign': 'top'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'center', 'backgroundColor': '#f8f8f8', 'height': '40px',
                        'maxHeight': '40px', 'lineHeight': '14px', 'overflow': 'hidden',
                        'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap', 'verticalAlign': 'top'
                    },
                    style_data_conditional=[
                        {'if': {'column_id': col}, 'width': '200px', 'maxWidth': '200px'}
                        for col in df_response.columns
                    ],
                    style_header_conditional=[
                        {'if': {'column_id': col}, 'width': '200px', 'maxWidth': '200px'}
                        for col in df_response.columns
                    ],
                    tooltip_data=tooltip_data,
                    # tooltip_header=header_tooltips,
                    tooltip_duration=None,
                    fill_width=False
                )

                query_section = None
                if query_text is not None:
                    formatted_sql = format_sql_query(query_text)
                    query_index = f"{len(chat_history)}-{len(current_messages)}"
                    query_section = html.Div([
                        html.Div([
                            html.Button([
                                html.Span("Show code", id={"type": "toggle-text", "index": query_index}, style={"display": "none"})
                            ],
                            id={"type": "toggle-query", "index": query_index},
                            className="toggle-query-button",
                            n_clicks=0)
                        ], className="toggle-query-container"),
                        html.Div([
                            html.Pre([
                                html.Code(formatted_sql, className="sql-code")
                            ], className="sql-pre")
                        ],
                        id={"type": "query-code", "index": query_index},
                        className="query-code-container hidden")
                    ], id={"type": "query-section", "index": query_index}, className="query-section")

                export_button = html.Button(
                    "Export as CSV",
                    id={"type": "export-button", "index": table_uuid},
                    className="insight-button",
                    style={'marginRight': '16px'}
                )

                insight_button = html.Button(
                    "Generate Insights",
                    id={"type": "insight-button", "index": table_uuid},
                    className="insight-button"
                )
                insight_output = dcc.Loading(
                    id={"type": "insight-loading", "index": table_uuid},
                    type="circle",
                    color="#000000",
                    children=html.Div(id={"type": "insight-output", "index": table_uuid})
                )

                content = html.Div([
                    html.Div(data_table, style={'marginBottom': '10px'}),
                    html.Div([export_button, insight_button], style={'display': 'flex'}),
                    insight_output
                ])

        # Create bot response
        bot_response = html.Div([
            html.Div([
                html.Div(className="model-avatar")
            ], className="model-info"),
            html.Div([
                content,
            ], className="message-content")
        ], className="bot-message message")

        # Update chat history with both user message and bot response
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = current_messages[:-1] + [bot_response]
        return current_messages[:-1] + [bot_response], chat_history, {"trigger": False, "message": ""} , False, new_conv_id

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again later."
        error_response = html.Div([
            html.Div([
                html.Div(className="model-avatar")
            ], className="model-info"),
            html.Div([
                html.Div(error_msg, className="message-text-bot")
            ], className="message-content")
        ], className="bot-message message")

        # Update chat history with both user message and error response
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = current_messages[:-1] + [error_response]

        return current_messages[:-1] + [error_response], chat_history, {"trigger": False, "message": ""}, False, new_conv_id

# Callback to handle CSV export
@app.callback(
    Output("download-dataframe-csv", "data"),
    Output("processed-export-clicks", "data"), # New output to update the store
    Input({"type": "export-button", "index": ALL}, "n_clicks"),
    State("chat-history-store", "data"),
    State("processed-export-clicks", "data"), # New state to read from the store
    prevent_initial_call=True,
)
def export_csv(n_clicks_list, chat_history, processed_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, no_update

    triggered_input = ctx.triggered[0]
    triggered_prop_id = triggered_input["prop_id"]
    triggered_value = triggered_input["value"]

    # Check if a specific export button was clicked and its n_clicks is greater than 0
    if "export-button" in triggered_prop_id and triggered_value is not None and triggered_value > 0:
        button_id_dict = json.loads(triggered_prop_id.split(".")[0])
        table_uuid = button_id_dict["index"]

        # Check if this button's click has already been processed for this n_clicks value
        if processed_clicks.get(table_uuid) == triggered_value:
            return dash.no_update, no_update # Already processed this click

        df = None
        if chat_history:
            for session in chat_history:
                if 'dataframes' in session and table_uuid in session['dataframes']:
                    df_json = session['dataframes'][table_uuid]
                    df = pd.read_json(StringIO(df_json), orient='split')
                    break
    
        if df is None:
            return dash.no_update, no_update

        # Mark this click as processed
        processed_clicks[table_uuid] = triggered_value
        return dcc.send_data_frame(df.to_csv, f"exported_data_{table_uuid[:8]}.csv", index=False), processed_clicks
    return dash.no_update, no_update


# Toggle sidebar and speech button
@app.callback(
    [Output("sidebar", "className"),
     Output("new-chat-button", "style"),
     Output("sidebar-new-chat-button", "style"),
     Output("change-space-button", "style"),
     Output("logo-container", "className"),
     Output("nav-left", "className"),
     Output("left-component", "className"),
     Output("main-content", "className")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className"),
     State("left-component", "className"),
     State("main-content", "className")]
)
def toggle_sidebar(n_clicks, current_sidebar_class, current_left_component_class, current_main_content_class):
    if n_clicks:
        if "sidebar-open" in current_sidebar_class:
            # Sidebar is closing
            return "sidebar", {"display": "flex"}, {"display": "none"}, {"display": "flex"}, "logo-container", "nav-left", "left-component", "main-content"
        else:
            # Sidebar is opening
            return "sidebar sidebar-open", {"display": "none"}, {"display": "flex"}, {"display": "none"}, "logo-container logo-container-open", "nav-left nav-left-open", "left-component left-component-open", "main-content main-content-shifted"
    # Initial state
    return current_sidebar_class, {"display": "flex"}, {"display": "none"}, {"display": "flex"}, "logo-container", "nav-left", "left-component", current_main_content_class

# Add callback for chat item selection
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input({"type": "chat-item", "index": ALL}, "n_clicks")],
    [State("chat-history-store", "data"),
     State("chat-list", "children"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def show_chat_history(n_clicks, chat_history, current_chat_list, session_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Get the clicked item index
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    clicked_index = json.loads(triggered_id)["index"]

    if not chat_history or clicked_index >= len(chat_history):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Update session data to the clicked session
    new_session_data = {"current_session": clicked_index}

    # Update active state in chat list
    updated_chat_list = []
    for i, item in enumerate(current_chat_list):
        new_class = "chat-item active" if i == clicked_index else "chat-item"
        updated_chat_list.append(
            html.Div(
                item["props"]["children"],
                className=new_class,
                id={"type": "chat-item", "index": i}
            )
        )

    return (chat_history[clicked_index]["messages"],
            "welcome-container hidden",
            updated_chat_list,
            new_session_data)

# Modify the clientside callback to target the chat-container
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

# It now only resets the chat state, not the selected space.
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("conversation-id-store", "data", allow_duplicate=True)],
    [Input("new-chat-button", "n_clicks"),
     Input("sidebar-new-chat-button", "n_clicks")],
    [State("chat-messages", "children"),
     State("chat-trigger", "data"),
     State("chat-history-store", "data"),
     State("chat-list", "children"),
     State("query-running-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def reset_to_welcome(n_clicks1, n_clicks2, chat_messages, chat_trigger, chat_history_store,
                    chat_list, query_running, session_data):
    # Reset session when starting a new chat
    new_session_data = {"current_session": None}
    return ([], {"trigger": False, "message": ""},
            False, chat_history_store, new_session_data, None)

@app.callback(
    [
        Output("selected-space-id", "data", allow_duplicate=True),
        Output("chat-messages", "children", allow_duplicate=True),
        Output("chat-trigger", "data", allow_duplicate=True),
        Output("query-running-store", "data", allow_duplicate=True),
        Output("chat-history-store", "data", allow_duplicate=True),
        Output("session-store", "data", allow_duplicate=True),
        Output("conversation-id-store", "data", allow_duplicate=True)
    ],
    Input("change-space-button", "n_clicks"),
    [
        State("chat-history-store", "data"),
    ],
    prevent_initial_call=True
)
def change_space_and_reset(n_clicks, chat_history):
    if not n_clicks:
        return [dash.no_update] * 7

    # This logic is from reset_to_welcome
    new_session_data = {"current_session": None}

    # Return tuple must have 8 values
    return (
        None,  # for selected-space-id -> shows overlay
        [],  # for chat-messages
        {"trigger": False, "message": ""},  # for chat-trigger
        False,  # for query-running-store
        chat_history,  # for chat-history-store (no change)
        new_session_data,  # for session-store
        None  # for conversation-id-store
    )

@app.callback(
    [Output("welcome-container", "className", allow_duplicate=True)],
    [Input("chat-messages", "children")],
    prevent_initial_call=True
)
def reset_query_running(chat_messages):
    # Return as a single-item list
    if chat_messages:
        return ["welcome-container hidden"]
    else:
        return ["welcome-container visible"]

# Add callback to disable input while query is running
@app.callback(
    [Output("chat-input-fixed", "disabled"),
     Output("mic-button", "disabled"),
     Output("send-button-fixed", "disabled"),
     Output("new-chat-button", "disabled"),
     Output("sidebar-new-chat-button", "disabled")],
    [Input("query-running-store", "data")],
    prevent_initial_call=True
)
def toggle_input_disabled(query_running):
    # Disable input and buttons when query is running
    return query_running, query_running, query_running, query_running, query_running

# Add callback for toggling SQL query visibility
@app.callback(
    [Output({"type": "query-code", "index": MATCH}, "className"),
     Output({"type": "toggle-text", "index": MATCH}, "children")],
    [Input({"type": "toggle-query", "index": MATCH}, "n_clicks")],
    prevent_initial_call=True
)
def toggle_query_visibility(n_clicks):
    if n_clicks % 2 == 1:
        return "query-code-container visible", "Hide code"
    return "query-code-container hidden", "Show code"

# Callback to open the insight prompt modal and store the DataFrame UUID
@app.callback(
    [Output("insight-prompt-modal", "is_open"),
     Output("current-dataframe-uuid", "data"),
     Output("processed-insight-clicks", "data")], # New output to update the store
    [Input({"type": "insight-button", "index": ALL}, "n_clicks")],
    State({"type": "insight-button", "index": ALL}, "id"),
    State("processed-insight-clicks", "data"), # New state to read from the store
    prevent_initial_call=True
)
def open_insight_modal(n_clicks_list, btn_ids_list, processed_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    triggered_input = ctx.triggered[0]
    triggered_prop_id = triggered_input["prop_id"]
    triggered_value = triggered_input["value"]

    # Check if a specific insight button was clicked and its n_clicks is greater than 0
    if "insight-button" in triggered_prop_id and triggered_value is not None and triggered_value > 0:
        triggered_btn_id = json.loads(triggered_prop_id.split(".")[0])
        table_uuid = triggered_btn_id['index']

        # Check if this button's click has already been processed for this n_clicks value
        if processed_clicks.get(table_uuid) == triggered_value:
            return no_update, no_update, no_update # Already processed this click

        # Mark this click as processed
        processed_clicks[table_uuid] = triggered_value
        return True, table_uuid, processed_clicks
    return no_update, no_update, no_update

# Callback to disable confirm button if textarea is empty
@app.callback(
    Output("confirm-insight-prompt-button", "disabled"),
    Input("insight-prompt-textarea", "value"),
    prevent_initial_call=False
)
def set_confirm_button_disabled(textarea_value):
    return not bool(textarea_value and textarea_value.strip())

# NEW CALLBACK: To close the modal immediately on confirm button click
@app.callback(
    Output("insight-prompt-modal", "is_open", allow_duplicate=True),
    Input("confirm-insight-prompt-button", "n_clicks"),
    prevent_initial_call=True
)
def close_modal_on_confirm(n_clicks):
    if n_clicks:
        return False # Close the modal immediately
    return no_update

# Callback to generate insights after prompt confirmation
@app.callback(
    [Output({"type": "insight-output", "index": ALL}, "children"),
     Output("insight-generation-status", "children"),
     Output("processed-confirm-insight-clicks", "data")], # Output for the new store
    [Input("confirm-insight-prompt-button", "n_clicks")],
    [State("insight-prompt-textarea", "value"),
     State("current-dataframe-uuid", "data"),
     State("chat-history-store", "data"),
     State("processed-confirm-insight-clicks", "data")], # State for the new store
    prevent_initial_call=True
)
def confirm_and_generate_insights(n_clicks, prompt_value, table_uuid, chat_history, processed_confirm_clicks):
    if not n_clicks:
        return no_update, no_update, no_update

    # Create a unique key for this specific confirmation action
    confirm_key = f"{table_uuid}-{n_clicks}"

    # If this confirm action has already been processed, prevent re-triggering
    if processed_confirm_clicks.get(confirm_key):
        return no_update, no_update, no_update

    insights_output_children = [no_update] * len(dash.callback_context.outputs_list[0])

    if not table_uuid:
        return insights_output_children, html.Div("Error: No data selected for insights.", style={"color": "red"}), no_update

    df = None
    if chat_history and len(chat_history) > 0:
        for session in chat_history:
            if 'dataframes' in session and table_uuid in session['dataframes']:
                df_json = session['dataframes'][table_uuid]
                df = pd.read_json(StringIO(df_json), orient='split')
                break

    if df is None:
        return insights_output_children, html.Div("Error: Data not found for insights.", style={"color": "red"}), no_update
    
    # Show loading message
    # This status message will be overwritten once insights are generated
    
    try:
        insights = call_llm_for_insights(df, prompt=prompt_value)
        
        # Find the correct insight output component to update
        for i, output_dict in enumerate(dash.callback_context.outputs_list[0]):
            if output_dict['id']['index'] == table_uuid:
                insights_output_children[i] = html.Div(
                    dcc.Markdown(insights),
                    style={"marginTop": "32px", "background": "#f4f4f4", "padding": "16px", "borderRadius": "4px"},
                    className="insight-output"
                )
                break
        
        # Mark this confirmation as processed
        processed_confirm_clicks[confirm_key] = True
        return insights_output_children, "", processed_confirm_clicks
    except Exception as e:
        error_message = f"Error generating insights: {str(e)}"
        return insights_output_children, html.Div(error_message, style={"color": "red"}), no_update

# Callback to fetch spaces on load
@app.callback(
    Output("spaces-list", "data"),
    Input("space-select-container", "id"),
    prevent_initial_call=False
)
def fetch_spaces(_):
    try:
        headers = request.headers
        token = headers.get('X-Forwarded-Access-Token')
        host = os.environ.get("DATABRICKS_HOST")
        client = GenieClient(host=host, space_id="", token=token)
        spaces = client.list_spaces()
        return spaces
    except Exception as e:
        return []

# Populate dropdown options
@app.callback(
    Output("space-dropdown", "options"),
    Input("spaces-list", "data"),
    prevent_initial_call=False
)
def update_space_dropdown(spaces):
    if not spaces:
        return [{"label": "No available agents", "value": "no_spaces_found", "disabled": True}]
    options = []
    for s in spaces:
        title = s.get('title', '')
        space_id = s.get('space_id', '')
        label_lines = [title]
        #label_lines.append(space_id)
        # label = " | ".join(label_lines)  # or use '\\n'.join(label_lines) for multi-line (but most browsers will show as a single line)
        options.append({"label": title, "value": space_id})
    return options


#Enable / Disable the "Explore Agent" button based on the dropdown value

@app.callback(
    Output("select-space-button", "disabled"),
    Input("space-dropdown", "value"),
    prevent_initial_call=False
)
def enable_select_space_button(selected_value ):
    if selected_value is None:
        return True
    return False

# Handle space selection
@app.callback(
    [Output("selected-space-id", "data", allow_duplicate=True),
     Output("space-select-container", "style"),
     Output("main-content", "style"),
     Output("welcome-title", "children", allow_duplicate=True), # Allow duplicate
     Output("welcome-description", "children", allow_duplicate=True)], # Allow duplicate
    Input("select-space-button", "n_clicks"),
    State("space-dropdown", "value"),
    State("spaces-list", "data"),
    prevent_initial_call=True
)
def select_space(n_clicks, space_id, spaces):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if not space_id:
        return dash.no_update, {"display": "flex", "flexDirection": "column", "alignItems": "start", "justifyContent": "center", "height": "100%"}, {"display": "none"}, dash.no_update, dash.no_update
    # Find the selected space's title and description
    selected = next((s for s in spaces if s["space_id"] == space_id), None)
    title = selected.get("title")
    description = selected.get("description")
    return space_id, {"display": "none"}, {"display": "block"}, title, description

# Add a callback to control visibility of main-content and space-select-container
@app.callback(
    [
        Output("main-content", "style", allow_duplicate=True),
        Output("space-select-container", "style", allow_duplicate=True),
        Output("top-nav", "style")
    ],
    Input("selected-space-id", "data"),
    prevent_initial_call=True
)
def toggle_main_ui(selected_space_id):
    if selected_space_id:
        # Main content view is active
        main_style = {"display": "block"}
        overlay_style = {"display": "none"}
        nav_top_style = {"display": "flex", "position": "fixed", "top": "0", "left": "0", "width": "100%", "zIndex": "1001"}
        return main_style, overlay_style, nav_top_style
    else:
        # Space selection overlay is active
        main_style = {"display": "none"}
        overlay_style = {"display": "flex"}
        nav_top_style = {"display": "none"}
        return main_style, overlay_style, nav_top_style
# Add clientside callback for scrolling to bottom of chat when insight is generated
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
                if (chatMessages.lastElementChild) {
                    chatMessages.lastElementChild.scrollIntoView({behavior: 'auto', block: 'end'});
                }
            }
        }, 100);
        return '';
    }
    """,
    Output('dummy-insight-scroll', 'children'),
    Input({'type': 'insight-output', 'index': ALL}, 'children'),
    prevent_initial_call=True
)

@app.callback(
    Output("selected-space-id", "data", allow_duplicate=True),
    [Input("logout-button-nav", "n_clicks"),
     Input("logout-button-overlay", "n_clicks")],
    prevent_initial_call=True
)
def logout_and_clear_space(n_clicks_nav, n_clicks_overlay):
    return None

# Add a callback to control the root-container style to prevent scrolling when overlay is visible
@app.callback(
    Output("root-container", "style"),
    Input("selected-space-id", "data"),
    prevent_initial_call=False
)
def set_root_style(selected_space_id):
    # root-container does not need specific height/overflow anymore as content inside has fixed top margin
    return {"height": "auto"}

# Add a callback to update the title based on spaces-list
# @app.callback(
#     Output("space-select-title", "children"),
#     Input("spaces-list", "data"),
#     prevent_initial_call=False
# )
# def update_space_select_title(spaces):
#     if not spaces:
#         return [html.Span(className="space-select-spinner"), "Loading Agents..."]
#     return "Select an Agent"

@app.callback(
    Output("query-tooltip", "className"),
    Input("query-running-store", "data"),
    prevent_initial_call=False
)
def update_query_tooltip_class(query_running):
    # Only show tooltip if query is running
    if query_running:
        return "query-tooltip query-tooltip-active"
    else:
        return "query-tooltip"

# Callback to fetch username and store it
@app.callback(
    Output("username-store", "data"),
    Input("root-container", "children"), # Trigger on initial load of the app
    prevent_initial_call=False
)
def fetch_username(_):
    try:
        username = request.headers.get("X-Forwarded-Preferred-Username", "").split("@")[0]
        username = username.split(".")
        username = [part[0].upper() + part[1:] for part in username]
        username = " ".join(username)
        return username
    except Exception as e:
        logger.error(f"Error fetching username: {e}")
        return None

# Callback to update the username display div
@app.callback(
    [Output("username-display-nav", "children"),
    Output("username-display-overlay", "children"),
    Output("welcome-user-greeting", "children")],
    Input("username-store", "data"),
    prevent_initial_call=False
)
def update_username_display(username):
    if username:
        first_name = username.split(" ")[0]
        greeting = f"Hello, {first_name}"
        return username, username, greeting

    return None, None, None

if __name__ == "__main__":
    app.run_server(debug=False)