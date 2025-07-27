import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context, no_update, clientside_callback, dash_table
import dash_bootstrap_components as dbc
import json
from genie_room import genie_query
import pandas as pd
import os
from dotenv import load_dotenv
import sqlparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
load_dotenv()

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Add default welcome text that can be customized
DEFAULT_WELCOME_TITLE = "Supply Chain Optimization"
DEFAULT_WELCOME_DESCRIPTION = "Analyze your Supply Chain Performance leveraging AI/BI Dashboard. Deep dive into your data and metrics."

# Add default suggestion questions
DEFAULT_SUGGESTIONS = [
    "What tables are there and how are they connected? Give me a short summary.",
    "Which distribution center has the highest chance of being a bottleneck?",
    "Explain the dataset",
    "What was the demand for our products by week in 2024?"
]

# Define the layout
app.layout = html.Div([
    # Top navigation bar
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
                ], id="new-chat-button", className="nav-button",disabled=False),
                html.Button([
                    html.Img(src="assets/plus_icon.svg", className="new-chat-icon"),
                    html.Div("New chat", className="new-chat-text")
                ], id="sidebar-new-chat-button", className="new-chat-button",disabled=False)
            ], id="nav-left", className="nav-left"),
            
            # Sidebar
            html.Div([
                html.Div([
                    html.Div("Your conversations with Genie", className="sidebar-header-text"),
                ], className="sidebar-header"),
                html.Div([], className="chat-list", id="chat-list")
            ], id="sidebar", className="sidebar")
        ], id="left-component", className="left-component"),
        
        html.Div([
            html.Div("Genie Space", id="logo-container", className="logo-container")
        ], className="nav-center"),
        html.Div([
            html.Div("Y", className="user-avatar"),
            html.A(
                html.Button(
                    "Logout",
                    id="logout-button",
                    className="logout-button"
                ),
                href=f"https://{os.getenv('DATABRICKS_HOST')}/login.html",
                className="logout-link"
            )
        ], className="nav-right")
    ], className="top-nav"),
    
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
                        html.Button([
                            html.Img(src="assets/settings_icon.svg", className="settings-icon"),
                            html.Div("Customize welcome message", className="button-tooltip")
                        ],
                        id="edit-welcome-button",
                        className="edit-welcome-button",
                        title="Customize welcome message")
                    ], className="welcome-title-container"),
                    
                    html.Div(id="welcome-description", 
                            className="welcome-message-description",
                            children=DEFAULT_WELCOME_DESCRIPTION),
                    
                    # Add modal for editing welcome text
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Customize Welcome Message")),
                        dbc.ModalBody([
                            html.Div([
                                html.Label("Welcome Title", className="modal-label"),
                                dbc.Input(
                                    id="welcome-title-input",
                                    type="text",
                                    placeholder="Enter a title for your welcome message",
                                    className="modal-input"
                                ),
                                html.Small(
                                    "This title appears at the top of your welcome screen",
                                    className="text-muted d-block mt-1"
                                )
                            ], className="modal-input-group"),
                            html.Div([
                                html.Label("Welcome Description", className="modal-label"),
                                dbc.Textarea(
                                    id="welcome-description-input",
                                    placeholder="Enter a description that helps users understand the purpose of your application",
                                    className="modal-input",
                                    style={"height": "80px"}
                                ),
                                html.Small(
                                    "This description appears below the title and helps guide your users",
                                    className="text-muted d-block mt-1"
                                )
                            ], className="modal-input-group"),
                            html.Div([
                                html.Label("Suggestion Questions", className="modal-label"),
                                html.Small(
                                    "Customize the four suggestion questions that appear on the welcome screen",
                                    className="text-muted d-block mb-3"
                                ),
                                dbc.Input(
                                    id="suggestion-1-input",
                                    type="text",
                                    placeholder="First suggestion question",
                                    className="modal-input mb-2"
                                ),
                                dbc.Input(
                                    id="suggestion-2-input",
                                    type="text",
                                    placeholder="Second suggestion question",
                                    className="modal-input mb-2"
                                ),
                                dbc.Input(
                                    id="suggestion-3-input",
                                    type="text",
                                    placeholder="Third suggestion question",
                                    className="modal-input mb-2"
                                ),
                                dbc.Input(
                                    id="suggestion-4-input",
                                    type="text",
                                    placeholder="Fourth suggestion question",
                                    className="modal-input"
                                )
                            ], className="modal-input-group")
                        ]),
                        dbc.ModalFooter([
                            dbc.Button(
                                "Cancel",
                                id="close-modal",
                                className="modal-button",
                                color="light"
                            ),
                            dbc.Button(
                                "Save Changes",
                                id="save-welcome-text",
                                className="modal-button-primary",
                                color="primary"
                            )
                        ])
                    ], id="edit-welcome-modal", is_open=False, size="lg", backdrop="static"),
                    
                    # Suggestion buttons with IDs
                    html.Div([
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div("What tables are there and how are they connected? Give me a short summary.", 
                                   className="suggestion-text", id="suggestion-1-text")
                        ], id="suggestion-1", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div("Which distribution center has the highest chance of being a bottleneck?",
                                   className="suggestion-text", id="suggestion-2-text")
                        ], id="suggestion-2", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div("Explain the dataset",
                                   className="suggestion-text", id="suggestion-3-text")
                        ], id="suggestion-3", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Div("What was the demand for our products by week in 2024?",
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
                            className="query-tooltip hidden")
                ], id="fixed-input-container", className="fixed-input-container"),
                html.Div("Always review the accuracy of responses.", className="disclaimer-fixed")
            ], id="fixed-input-wrapper", className="fixed-input-wrapper"),
        ], id="chat-container", className="chat-container"),
    ], id="main-content", className="main-content"),
    
    html.Div(id='dummy-output'),
    dcc.Store(id="chat-trigger", data={"trigger": False, "message": ""}),
    dcc.Store(id="chat-history-store", data=[]),
    dcc.Store(id="query-running-store", data=False),
    dcc.Store(id="session-store", data={"current_session": None})
])

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
            "You are a professional data analyst. Given the following table data, "
            "provide deep, actionable analysis for 1. Key insights and trends 2. Notable patterns and" 
            " anomalies 3. Business implications."
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
            html.Div("Y", className="user-avatar"),
            html.Span("You", className="model-name")
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
     Output("query-running-store", "data", allow_duplicate=True)],
    [Input("chat-trigger", "data")],
    [State("chat-messages", "children"),
     State("chat-history-store", "data")],
    prevent_initial_call=True
)
def get_model_response(trigger_data, current_messages, chat_history):
    if not trigger_data or not trigger_data.get("trigger"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    user_input = trigger_data.get("message", "")
    if not user_input:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        response, query_text = genie_query(user_input)
        
        if isinstance(response, str):
            content = dcc.Markdown(response, className="message-text")
        else:
            # Data table response
            df = pd.DataFrame(response)
            
            # Store the DataFrame in chat_history for later retrieval by insight button
            if chat_history and len(chat_history) > 0:
                chat_history[0].setdefault('dataframes', {})[f"table-{len(chat_history)}"] = df.to_json(orient='split')
            else:
                chat_history = [{"dataframes": {f"table-{len(chat_history)}": df.to_json(orient='split')}}]
            
            # Create the table with adjusted styles
            data_table = dash_table.DataTable(
                id=f"table-{len(chat_history)}",
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                
                # Export configuration
                export_format="csv",
                export_headers="display",
                
                # Other table properties
                page_size=10,
                style_table={
                    'display': 'inline-block',
                    'overflowX': 'auto',
                    'width': '95%',
                    'marginRight': '20px'
                },
                style_cell={
                    'textAlign': 'left',
                    'fontSize': '12px',
                    'padding': '4px 10px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont,Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif',
                    'backgroundColor': 'transparent',
                    'maxWidth': 'fit-content',
                    'minWidth': '100px'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'borderBottom': '1px solid #eaecef'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                fill_width=False,
                page_current=0,
                page_action='native'
            )

            # Format SQL query if available
            query_section = None
            if query_text is not None:
                formatted_sql = format_sql_query(query_text)
                query_index = f"{len(chat_history)}-{len(current_messages)}"
                
                query_section = html.Div([
                    html.Div([
                        html.Button([
                            html.Span("Show code", id={"type": "toggle-text", "index": query_index})
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
                id={"type": "insight-button", "index": f"table-{len(chat_history)}"},
                className="insight-button",
                style={"border": "none", "background": "#f0f0f0", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}
            )
            insight_output = dcc.Loading(
                id={"type": "insight-loading", "index": f"table-{len(chat_history)}"},
                type="circle",
                color="#000000",
                children=html.Div(id={"type": "insight-output", "index": f"table-{len(chat_history)}"})
            )

            # Create content with table and optional SQL section
            content = html.Div([
                html.Div([data_table], style={
                    'marginBottom': '20px',
                    'paddingRight': '5px'
                }),
                query_section if query_section else None,
                insight_button,
                insight_output,
            ])
        
        # Create bot response
        bot_response = html.Div([
            html.Div([
                html.Div(className="model-avatar"),
                html.Span("Genie", className="model-name")
            ], className="model-info"),
            html.Div([
                content,
                html.Div([
                    html.Div([
                        html.Button(
                            id={"type": "thumbs-up-button", "index": len(chat_history)},
                            className="thumbs-up-button"
                        ),
                        html.Button(
                            id={"type": "thumbs-down-button", "index": len(chat_history)},
                            className="thumbs-down-button"
                        )
                    ], className="message-actions")
                ], className="message-footer")
            ], className="message-content")
        ], className="bot-message message")
        
        # Update chat history with both user message and bot response
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = current_messages[:-1] + [bot_response]  
        return current_messages[:-1] + [bot_response], chat_history, {"trigger": False, "message": ""}, False
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again later."
        error_response = html.Div([
            html.Div([
                html.Div(className="model-avatar"),
                html.Span("Genie", className="model-name")
            ], className="model-info"),
            html.Div([
                html.Div(error_msg, className="message-text")
            ], className="message-content")
        ], className="bot-message message")
        
        # Update chat history with both user message and error response
        if chat_history and len(chat_history) > 0:
            chat_history[0]["messages"] = current_messages[:-1] + [error_response]
        
        return current_messages[:-1] + [error_response], chat_history, {"trigger": False, "message": ""}, False

# Toggle sidebar and speech button
@app.callback(
    [Output("sidebar", "className"),
     Output("new-chat-button", "style"),
     Output("sidebar-new-chat-button", "style"),
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
            return "sidebar", {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", "main-content"
        else:
            # Sidebar is opening
            return "sidebar sidebar-open", {"display": "none"}, {"display": "flex"}, "logo-container logo-container-open", "nav-left nav-left-open", "left-component left-component-open", "main-content main-content-shifted"
    # Initial state
    return current_sidebar_class, {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", current_main_content_class

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
        var chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    Input('chat-messages', 'children'),
    prevent_initial_call=True
)

# Modify the new chat button callback to reset session
@app.callback(
    [Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("query-running-store", "data", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
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
            False, chat_history_store, new_session_data)

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
     Output("sidebar-new-chat-button", "disabled"),
     Output("query-tooltip", "className")],
    [Input("query-running-store", "data")],
    prevent_initial_call=True
)
def toggle_input_disabled(query_running):
    # Show tooltip when query is running, hide it otherwise
    tooltip_class = "query-tooltip visible" if query_running else "query-tooltip hidden"
    
    # Disable input and buttons when query is running
    return query_running, query_running, query_running, query_running, tooltip_class


# Fix the callback for thumbs up/down buttons
@app.callback(
    [Output({"type": "thumbs-up-button", "index": MATCH}, "className"),
     Output({"type": "thumbs-down-button", "index": MATCH}, "className")],
    [Input({"type": "thumbs-up-button", "index": MATCH}, "n_clicks"),
     Input({"type": "thumbs-down-button", "index": MATCH}, "n_clicks")],
    [State({"type": "thumbs-up-button", "index": MATCH}, "className"),
     State({"type": "thumbs-down-button", "index": MATCH}, "className")],
    prevent_initial_call=True
)
def handle_feedback(up_clicks, down_clicks, up_class, down_class):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_type = json.loads(trigger_id)["type"]
    
    if button_type == "thumbs-up-button":
        new_up_class = "thumbs-up-button active" if "active" not in up_class else "thumbs-up-button"
        new_down_class = "thumbs-down-button"
    else:
        new_up_class = "thumbs-up-button"
        new_down_class = "thumbs-down-button active" if "active" not in down_class else "thumbs-down-button"
    
    return new_up_class, new_down_class

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

# Add callbacks for welcome text customization
@app.callback(
    [Output("edit-welcome-modal", "is_open", allow_duplicate=True),
     Output("welcome-title-input", "value"),
     Output("welcome-description-input", "value"),
     Output("suggestion-1-input", "value"),
     Output("suggestion-2-input", "value"),
     Output("suggestion-3-input", "value"),
     Output("suggestion-4-input", "value")],
    [Input("edit-welcome-button", "n_clicks")],
    [State("welcome-title", "children"),
     State("welcome-description", "children"),
     State("suggestion-1-text", "children"),
     State("suggestion-2-text", "children"),
     State("suggestion-3-text", "children"),
     State("suggestion-4-text", "children")],
    prevent_initial_call=True
)
def open_modal(n_clicks, current_title, current_description, s1, s2, s3, s4):
    if not n_clicks:
        return [no_update] * 7
    return True, current_title, current_description, s1, s2, s3, s4

@app.callback(
    [Output("welcome-title", "children", allow_duplicate=True),
     Output("welcome-description", "children", allow_duplicate=True),
     Output("suggestion-1-text", "children", allow_duplicate=True),
     Output("suggestion-2-text", "children", allow_duplicate=True),
     Output("suggestion-3-text", "children", allow_duplicate=True),
     Output("suggestion-4-text", "children", allow_duplicate=True),
     Output("edit-welcome-modal", "is_open", allow_duplicate=True)],
    [Input("save-welcome-text", "n_clicks"),
     Input("close-modal", "n_clicks")],
    [State("welcome-title-input", "value"),
     State("welcome-description-input", "value"),
     State("suggestion-1-input", "value"),
     State("suggestion-2-input", "value"),
     State("suggestion-3-input", "value"),
     State("suggestion-4-input", "value"),
     State("welcome-title", "children"),
     State("welcome-description", "children"),
     State("suggestion-1-text", "children"),
     State("suggestion-2-text", "children"),
     State("suggestion-3-text", "children"),
     State("suggestion-4-text", "children")],
    prevent_initial_call=True
)
def handle_modal_actions(save_clicks, close_clicks,
                        new_title, new_description, s1, s2, s3, s4,
                        current_title, current_description,
                        current_s1, current_s2, current_s3, current_s4):
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * 7

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "close-modal":
        return [current_title, current_description, 
                current_s1, current_s2, current_s3, current_s4, False]
    elif trigger_id == "save-welcome-text":
        # Save the changes
        title = new_title if new_title else DEFAULT_WELCOME_TITLE
        description = new_description if new_description else DEFAULT_WELCOME_DESCRIPTION
        suggestions = [
            s1 if s1 else DEFAULT_SUGGESTIONS[0],
            s2 if s2 else DEFAULT_SUGGESTIONS[1],
            s3 if s3 else DEFAULT_SUGGESTIONS[2],
            s4 if s4 else DEFAULT_SUGGESTIONS[3]
        ]
        return [title, description, *suggestions, False]

    return [no_update] * 7

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
        return None  # Don't show anything before click
    table_id = btn_id["index"]
    # Retrieve the DataFrame from chat_history
    df = None
    if chat_history and len(chat_history) > 0:
        df_json = chat_history[0].get('dataframes', {}).get(table_id)
        if df_json:
            df = pd.read_json(df_json, orient='split')
    if df is None:
        return None
    insights = call_llm_for_insights(df)
    return html.Div([
        html.Div([
            dcc.Markdown(insights, className="insight-content")
        ], className="insight-body")
    ], className="insight-wrapper")

if __name__ == "__main__":
    app.run_server(debug=True)
