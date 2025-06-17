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

# Add default welcome text that can be customized
DEFAULT_WELCOME_TITLE = "Welcome to Your Data Assistant"
DEFAULT_WELCOME_DESCRIPTION = "Explore and analyze your data with AI-powered insights. Ask questions, discover trends, and make data-driven decisions."

# Define the layout
app.layout = html.Div([
    html.Div([
        dcc.Store(id="selected-space-id", data=None),
        dcc.Store(id="spaces-list", data=[]),
        dcc.Store(id="conversation-id-store", data=None),
        dcc.Store(id="username-store", data=None),
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

                ], id="nav-left", className="nav-left", style={"display": "none"}), # Initially hidden

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
                    html.Div(className="company-logo")
                    ], id="logo-container",
                    className="logo-container"
                )
            ], id="nav-center", className="nav-center", style={"display": "none"}), # Initially hidden
            html.Div([
                html.Div(className="user-avatar"),
                html.Div(
                    id="username-display",
                    style={'color': 'black', 'fontSize': '1em'}
                ),
                html.A(
                    html.Button([html.Img(src="assets/logout_icon.svg")],
                        id="logout-button",
                        className="logout-button",
                        title="Logout"
                    ),
                    href=f"https://{os.getenv('DATABRICKS_HOST')}/login.html",
                    className="logout-link"
                )
            ], className="nav-right")
        ], className="top-nav", style={"position": "fixed", "top": "0", "left": "0", "width": "100%", "zIndex": "1001"}), # Added style for fixed header

        # Main content wrapper (includes space overlay and main chat)
        # This wrapper will have a margin-top equal to the fixed header's height
        html.Div([
            # Space selection overlay - now positioned relative to this new wrapper's top
            html.Div([
                html.Div([
                    html.Div(className="company-logo"),
                    html.H1("BI Agent Platform", className="main-title"),
                    html.Div([
                        html.Span(className="space-select-spinner"),
                        "Loading Agents..."
                    ], id="space-select-title", className="space-select-title"),
                    dcc.Dropdown(id="space-dropdown", options=[], placeholder="Choose an Agent", className="space-select-dropdown", optionHeight=40, searchable=True),
                    html.Button("Explore Agent", id="select-space-button", className="space-select-button"),
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
                                html.Div(id="welcome-title", className="welcome-message", children=DEFAULT_WELCOME_TITLE),
                            ], className="welcome-title-container"),

                            html.Div(id="welcome-description",
                                    className="welcome-message-description",
                                    children=DEFAULT_WELCOME_DESCRIPTION),

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
        ], style={"marginTop": "60px", "height": "calc(100vh - 60px)", "position": "relative"}), # New wrapper for main content and overlay

        html.Div(id='dummy-output'),
        dcc.Store(id="chat-trigger", data={"trigger": False, "message": ""}),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Store(id="query-running-store", data=False),
        dcc.Store(id="session-store", data={"current_session": None}),
        html.Div(id='dummy-insight-scroll')
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
        html.Div([
            html.Div(className="user-avatar")
        ], className="user-info"),
        html.Div(user_input, className="message-text")
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
                chat_history = [{"dataframes": {table_uuid: df.to_json(orient='split')}}]
        else:
            table_uuid = None # No df to store

        if isinstance(response, str):
            response = response.replace("`", "")
            content = dcc.Markdown(response, className="message-text", style={"fontFamily": "Segoe UI", "fontSize": "14px"})
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
                                html.Span("Show code", id={"type": "toggle-text", "index": query_index}, style={"fontFamily": "Segoe UI", "display": "none"})
                            ], id={"type": "toggle-query", "index": query_index}, className="toggle-query-button", n_clicks=0)
                        ], className="toggle-query-container"),
                        html.Div([
                            html.Pre([html.Code(formatted_sql, className="sql-code", style={"fontFamily": "Segoe UI"})], className="sql-pre")
                        ], id={"type": "query-code", "index": query_index}, className="query-code-container hidden")
                    ], id={"type": "query-section", "index": query_index}, className="query-section")

                content = html.Div([
                    dcc.Markdown(markdown_response, style={"fontFamily": "Segoe UI", "fontSize": "14px"}),
                    query_section
                ]) if query_section else dcc.Markdown(markdown_response, style={"fontFamily": "Segoe UI", "fontSize": "14px"})

            else:
                table_data = df_response.to_dict('records')
                table_columns = [{"name": col, "id": col} for col in df_response.columns]

                tooltip_data = [
                    {col: {'value': str(row[col]), 'type': 'markdown'} for col in df_response.columns}
                    for row in table_data
                ]
                header_tooltips = {col: {'value': col, 'type': 'markdown'} for col in df_response.columns}

                data_table = dash_table.DataTable(
                    id=f"table-{len(chat_history)}",
                    data=table_data,
                    columns=table_columns,
                    export_format="csv",
                    export_headers="display",
                    style_table={'maxHeight': '300px', 'overflowY': 'auto', 'overflowX': 'auto', 'width': '95%'},
                    style_data={
                        'fontFamily': 'Segoe UI', 'fontSize': '14px', 'textAlign': 'left',
                        'padding': '5px', 'height': '40px', 'maxHeight': '40px',
                        'lineHeight': '14px', 'overflow': 'hidden', 'textOverflow': 'ellipsis',
                        'whiteSpace': 'nowrap', 'verticalAlign': 'top'
                    },
                    style_header={
                        'fontFamily': 'Segoe UI', 'fontSize': '14px', 'fontWeight': 'bold',
                        'textAlign': 'left', 'backgroundColor': '#f8f8f8', 'height': '40px',
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
                    tooltip_header=header_tooltips,
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
                                html.Span("Show code", id={"type": "toggle-text", "index": query_index}, style={"fontFamily": "Segoe UI", "display": "none"})
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

                insight_button = html.Button(
                    "Generate Insights",
                    id={"type": "insight-button", "index": table_uuid},
                    className="insight-button",
                    style={"border": "none", "background": "#f0f0f0", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer", "marginTop": "10px"}
                )
                insight_output = dcc.Loading(
                    id={"type": "insight-loading", "index": table_uuid},
                    type="circle",
                    color="#000000",
                    children=html.Div(id={"type": "insight-output", "index": table_uuid})
                )

                content = html.Div([
                    html.Div(data_table, style={'marginBottom': '10px'}),
                    query_section if query_section else None,
                    insight_button,
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
                html.Div(error_msg, className="message-text")
            ], className="message-content")
        ], className="bot-message message")

        # Update chat history with both user message and error response
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = current_messages[:-1] + [error_response]

        return current_messages[:-1] + [error_response], chat_history, {"trigger": False, "message": ""}, False, new_conv_id


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
    [Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-messages", "children", allow_duplicate=True),
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
    return ("welcome-container visible", [], {"trigger": False, "message": ""},
            False, chat_history_store, new_session_data, None)

@app.callback(
    [
        Output("selected-space-id", "data", allow_duplicate=True),
        Output("welcome-container", "className", allow_duplicate=True),
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
        return [dash.no_update] * 8

    # This logic is from reset_to_welcome
    new_session_data = {"current_session": None}

    # Return tuple must have 8 values
    return (
        None,  # for selected-space-id -> shows overlay
        "welcome-container visible",  # for welcome-container
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
     Output("send-button-fixed", "disabled"),
     Output("new-chat-button", "disabled"),
     Output("sidebar-new-chat-button", "disabled")],
    [Input("query-running-store", "data")],
    prevent_initial_call=True
)
def toggle_input_disabled(query_running):
    # Disable input and buttons when query is running
    return query_running, query_running, query_running, query_running

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

# Add callback for insight button
@app.callback(
    Output({"type": "insight-output", "index": dash.dependencies.MATCH}, "children"),
    Input({"type": "insight-button", "index": dash.dependencies.MATCH}, "n_clicks"),
    State({"type": "insight-button", "index": dash.dependencies.MATCH}, "id"),
    State("chat-history-store", "data"),
    prevent_initial_call=True
)
def generate_insights(n_clicks, btn_id, chat_history):
    if not n_clicks:
        return None
    table_id = btn_id["index"]
    df = None
    if chat_history and len(chat_history) > 0:
        df_json = chat_history[0].get('dataframes', {}).get(table_id)
        if df_json:
            df = pd.read_json(StringIO(df_json), orient='split')
    if df is None:
        return html.Div("No data available for insights.", style={"color": "red"})
    insights = call_llm_for_insights(df)
    return html.Div(
        dcc.Markdown(insights),
        style={"marginTop": "32px", "background": "#f4f4f4", "padding": "16px", "borderRadius": "4px"},
        className="insight-output"
    )

# Callback to fetch spaces on load
@app.callback(
    Output("spaces-list", "data"),
    Input("space-select-container", "id"),
    prevent_initial_call=False
)
def fetch_spaces(_):
    try:
        headers = request.headers
        # token = os.environ.get("DATABRICKS_TOKEN")
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
        return []
    options = []
    for s in spaces:
        title = s.get('title', '')
        space_id = s.get('space_id', '')
        label_lines = [title]
        #label_lines.append(space_id)
        # label = " | ".join(label_lines)  # or use '\\n'.join(label_lines) for multi-line (but most browsers will show as a single line)
        options.append({"label": title, "value": space_id})
    return options

# Handle space selection
@app.callback(
    [Output("selected-space-id", "data", allow_duplicate=True),
     Output("space-select-container", "style"),
     Output("main-content", "style"),
     Output("space-select-error", "children"),
     Output("welcome-title", "children", allow_duplicate=True), # Allow duplicate
     Output("welcome-description", "children", allow_duplicate=True)], # Allow duplicate
    Input("select-space-button", "n_clicks"),
    State("space-dropdown", "value"),
    State("spaces-list", "data"),
    prevent_initial_call=True
)
def select_space(n_clicks, space_id, spaces):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if not space_id:
        return dash.no_update, {"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center", "height": "100%"}, {"display": "none"}, "Please select an Agent.", dash.no_update, dash.no_update
    # Find the selected space's title and description
    selected = next((s for s in spaces if s["space_id"] == space_id), None)
    title = selected["title"] if selected and selected.get("title") else DEFAULT_WELCOME_TITLE
    description = selected["description"] if selected and selected.get("description") else DEFAULT_WELCOME_DESCRIPTION
    return space_id, {"display": "none"}, {"display": "block"}, "", title, description

# New callback to update welcome title and description on load or space change
@app.callback(
    [Output("welcome-title", "children", allow_duplicate=True),
     Output("welcome-description", "children", allow_duplicate=True)],
    [Input("selected-space-id", "data"),
     Input("spaces-list", "data")],
    prevent_initial_call=True
)
def update_welcome_content_on_load(selected_space_id, spaces):
    if not selected_space_id or not spaces:
        return DEFAULT_WELCOME_TITLE, DEFAULT_WELCOME_DESCRIPTION

    selected = next((s for s in spaces if s["space_id"] == selected_space_id), None)
    if selected:
        title = selected.get("title", DEFAULT_WELCOME_TITLE)
        description = selected.get("description", DEFAULT_WELCOME_DESCRIPTION)
        return title, description

    return DEFAULT_WELCOME_TITLE, DEFAULT_WELCOME_DESCRIPTION

# Add a callback to control visibility of main-content and space-select-container
@app.callback(
    [
        Output("main-content", "style", allow_duplicate=True),
        Output("space-select-container", "style", allow_duplicate=True),
        Output("left-component", "style"),
        Output("nav-center", "style"),
        Output("nav-left", "style")
    ],
    Input("selected-space-id", "data"),
    prevent_initial_call=True
)
def toggle_main_ui(selected_space_id):
    if selected_space_id:
        # Main content view is active
        main_style = {"display": "block"}
        overlay_style = {"display": "none"}
        # Ensure nav components are visible in main content area
        left_component_style = {"display": "flex"}
        center_nav_style = {"display": "flex"}
        nav_left_style = {"display": "flex"}
        return main_style, overlay_style, left_component_style, center_nav_style, nav_left_style
    else:
        # Space selection overlay is active
        main_style = {"display": "none"}
        overlay_style = {"display": "flex"}
        # Hide nav components
        left_component_style = {"display": "none"}
        center_nav_style = {"display": "none"}
        nav_left_style = {"display": "none"}
        return main_style, overlay_style, left_component_style, center_nav_style, nav_left_style
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
    Input({'type': 'insight-output', 'index': dash.dependencies.ALL}, 'children'),
    prevent_initial_call=True
)

@app.callback(
    Output("selected-space-id", "data", allow_duplicate=True),
    Input("logout-button", "n_clicks"),
    prevent_initial_call=True
)
def logout_and_clear_space(n_clicks):
    if n_clicks:
        return None
    return dash.no_update

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
@app.callback(
    Output("space-select-title", "children"),
    Input("spaces-list", "data"),
    prevent_initial_call=False
)
def update_space_select_title(spaces):
    if not spaces:
        return [html.Span(className="space-select-spinner"), "Loading Agents..."]
    return "Select an Agent"

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
    Output("username-display", "children"),
    Input("username-store", "data"),
    prevent_initial_call=False
)
def update_username_display(username):
    return username

if __name__ == "__main__":
    app.run_server(debug=False)