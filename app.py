import os
import json
import uuid
import re
import string
import importlib.util
from glob import glob
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from openai import OpenAI
from werkzeug.exceptions import HTTPException

app = Flask(__name__,static_folder='./dist',template_folder='./dist')

# --- DIRECTORY SETUP ---
DIRS = ["chats", "pages", "tools"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "main_ai": {
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4-turbo",
        "temperature": 0.7,
        "api_key": "",
        "instruction": "You are Main AI. You are a helpful assistant. Use tools whenever necessary to fulfill the user's request."
    },
    "code_ai": {
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4-turbo",
        "temperature": 0.2,
        "api_key": ""
    }
}

if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=4)

# --- HELPER FUNCTIONS ---
def get_settings():
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def is_safe_filename(name):
    avl_chars = string.ascii_letters + string.digits + " -_()"
    for c in name:
        if c not in avl_chars:
            return False
    return True

def sanitize_filename(name):
    avl_chars = string.ascii_letters + string.digits + " -_()"
    final_name = ""
    for c in name:
        if c in avl_chars:
            final_name += c
    
    if final_name == "":
        final_name = f"tool_{uuid.uuid4().hex[:8]}"

    return final_name

def get_client(ai_type="main_ai"):
    settings = get_settings()[ai_type]
    return OpenAI(
        api_key=settings["api_key"] or "dummy-key",
        base_url=settings["base_url"]
    )

def get_available_tools():
    tools = []
    for schema_file in glob("tools/*.json"):
        with open(schema_file, "r") as f:
            tools.append({"type": "function", "function": json.load(f)})
    return tools

def execute_tool(name, arguments):
    script_path = f"tools/{name}.py"
    files = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-3:] == '.py']
    if f"{name}.py" not in files:
        return f"Error: Tool {name} script not found."
    
    try:
        spec = importlib.util.spec_from_file_location(name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args_dict = json.loads(arguments)
        result = module.run(**args_dict)
        return json.dumps(result)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"

# --- SETTINGS API ---
@app.route('/api/settings', methods=['GET'])
def read_settings():
    return jsonify(get_settings())

@app.route('/api/settings', methods=['PUT'])
def update_settings():
    settings = get_settings()
    data = request.json
    
    if "main_ai" in data:
        settings["main_ai"].update(data["main_ai"])
    if "code_ai" in data:
        settings["code_ai"].update(data["code_ai"])
        
    save_settings(settings)
    return jsonify(settings)

# --- CHATS API ---
@app.route('/api/chats', methods=['GET'])
def get_chat_list():
    chats = []
    for chat_file in glob("chats/*.json"):
        with open(chat_file, "r") as f:
            data = json.load(f)
            chats.append({"id": data["id"], "title": data.get("title", "New Chat")})
    return jsonify(chats)

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat_content(chat_id):
    path = f"chats/{chat_id}.json"
    files = [f for f in os.listdir('chats') if os.path.isfile(os.path.join('chats', f)) and f[-5:] == '.json']
    if f"{chat_id}.json" not in files:
        return jsonify({"error": "Chat not found"}), 404
    with open(path, "r") as f:
        return jsonify(json.load(f))

@app.route('/api/chats', methods=['POST'])
def create_chat():
    chat_id = str(uuid.uuid4())
    chat_data = {"id": chat_id, "title": "New Chat", "messages": []}
    with open(f"chats/{chat_id}.json", "w") as f:
        json.dump(chat_data, f)
    return jsonify(chat_data)

@app.route('/api/chats/<chat_id>/message', methods=['POST'])
def send_message(chat_id):
    data = request.json
    user_message = data.get("message")
    
    path = f"chats/{chat_id}.json"
    files = [f for f in os.listdir('chats') if os.path.isfile(os.path.join('chats', f)) and f[-5:] == '.json']
    if f"{chat_id}.json" not in files:
        return jsonify({"error": "Chat not found"}), 404
        
    with open(path, "r") as f:
        chat_data = json.load(f)

    # Set title if it's the first message
    if not chat_data["messages"]:
        chat_data["title"] = user_message[:30]

    if user_message:
        chat_data["messages"].append({"role": "user", "content": user_message})

    settings = get_settings()["main_ai"]
    client = get_client("main_ai")
    
    messages = [{"role": "system", "content": settings["instruction"]}] + chat_data["messages"]
    available_tools = get_available_tools()
    
    # Tool execution loop
    while True:
        kwargs = {
            "model": settings["model_name"],
            "messages": messages,
            "temperature": settings["temperature"],
        }
        if available_tools:
            kwargs["tools"] = available_tools
            
        response = client.chat.completions.create(**kwargs)
        response_message = response.choices[0].message # check here
        
        chat_data["messages"].append(response_message.model_dump(exclude_none=True))
        messages.append(response_message.model_dump(exclude_none=True)) # check here

        if not response_message.tool_calls:
            break
            
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            function_response = execute_tool(function_name, function_args)
            
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response,
            }
            chat_data["messages"].append(tool_msg)
            messages.append(tool_msg)

    with open(path, "w") as f:
        json.dump(chat_data, f, indent=4)
        
    return jsonify(chat_data)

@app.route('/api/chats/<chat_id>/message', methods=['PUT'])
def edit_last_message(chat_id):
    path = f"chats/{chat_id}.json"
    new_message = request.json.get("message")

    files = [f for f in os.listdir('chats') if os.path.isfile(os.path.join('chats', f)) and f[-5:] == '.json']
    if f"{chat_id}.json" not in files:
        return jsonify({"error": "Chat not found"}), 404
    
    with open(path, "r") as f:
        chat_data = json.load(f)
        
    # Remove everything after and including the last user message
    for i in range(len(chat_data["messages"]) - 1, -1, -1):
        if chat_data["messages"][i]["role"] == "user":
            chat_data["messages"] = chat_data["messages"][:i]
            break
            
    with open(path, "w") as f:
        json.dump(chat_data, f, indent=4)
        
    # Re-trigger send message
    request.json["message"] = new_message
    return send_message(chat_id)

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    path = f"chats/{chat_id}.json"
    files = [f for f in os.listdir('chats') if os.path.isfile(os.path.join('chats', f)) and f[-5:] == '.json']
    if f"{chat_id}.json" not in files:
        return jsonify({"error": "Chat not found"}), 404

    os.remove(path)
    return jsonify({"status": "deleted"})

# --- TOOLS API ---
TOOL_INSTRUCTION = """You are an AI tool creator. When asked to create a tool, respond with EXACTLY two code blocks and no conversational text:
Block 1: A JSON block containing the tool's schema definition.
Block 2: A Python block containing the executable script with a run(**kwargs) entrypoint.

Example format:
```JSON
{
    "name": "tool_name",
    "description": "Tool description",
    "parameters": {
        "type": "object",
        "properties": {
            "argument1": {
                "type": "argument data type",
                "description": "argument 1 description"
            },
            "argument2": {
                "type": "argument data type",
                "description": "argument 2 description"
            }
        },
        "required": [
            "argument1"
        ]
    }
}
```

```python
def run(**kwargs):
    # Library imported inside function

    # Python implementation

    # Return the output if exists
    return 1
```"""

@app.route('/api/tools', methods=['GET'])
def get_tools():
    tools = []
    for schema_file in glob("tools/*.json"):
        with open(schema_file, "r") as f:
            data = json.load(f)
            tools.append({"name": data["name"], "description": data.get("description", "")})
    return jsonify(tools)

@app.route('/api/tools/<name>', methods=['GET'])
def get_tool_content(name):
    path = f"tools/{name}"
    files_json = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-5:] == '.json']
    files_py = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-3:] == '.py']
    if f"{name}.json" not in files_json or f"{name}.py" not in files_py:
        return jsonify({"error": "Tool not found"}), 404
    
    try:
        with open(f"tools/{name}.json", "r") as f:
            schema = json.load(f)
        with open(f"tools/{name}.py", "r") as f:
            code = f.read()
        return jsonify({"schema": schema, "code": code})
    except FileNotFoundError:
        return jsonify({"error": "Tool not found"}), 404

@app.route('/api/tools', methods=['POST'])
def generate_tool():
    prompt = request.json.get("prompt")
    existing_name = request.json.get("existing_name") # If editing
    
    client = get_client("code_ai")
    settings = get_settings()["code_ai"]
    
    messages = [{"role": "system", "content": TOOL_INSTRUCTION}]
    
    if existing_name:
        files_json = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-5:] == '.json']
        files_py = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-3:] == '.py']
        if f"{existing_name}.json" not in files_json or f"{existing_name}.py" not in files_py:
            return jsonify({"error": "Tool not found"}), 404

        try:
            with open(f"tools/{existing_name}.json", "r") as f:
                schema = f.read()
            with open(f"tools/{existing_name}.py", "r") as f:
                code = f.read()
            messages.append({"role": "user", "content": f"Current Schema:\n```json\n{schema}\n```\n\nCurrent Code:\n```python\n{code}\n```\n\nEdit instruction: {prompt}"})
        except:
            pass
    else:
        messages.append({"role": "user", "content": prompt})
        
    response = client.chat.completions.create(
        model=settings["model_name"],
        messages=messages,
        temperature=settings["temperature"]
    )
    
    reply = response.choices[0].message.content
    
    # Extract JSON and Python blocks
    json_match = re.search(r'```(?:json|JSON)\n(.*?)\n```', reply, re.DOTALL)
    py_match = re.search(r'```(?:python|Python)\n(.*?)\n```', reply, re.DOTALL)
    
    if not json_match or not py_match:
        return jsonify({"error": "AI failed to format code blocks correctly", "raw": reply}), 400
        
    schema_data = json.loads(json_match.group(1))
    tool_name = schema_data.get("name", f"tool_{uuid.uuid4().hex[:8]}")
    tool_name = sanitize_filename(tool_name)
    
    with open(f"tools/{tool_name}.json", "w") as f:
        json.dump(schema_data, f, indent=4)
    with open(f"tools/{tool_name}.py", "w") as f:
        f.write(py_match.group(1))
        
    # Cleanup old files if name changed during edit
    if existing_name and existing_name != tool_name:
        delete_tool(existing_name)
        
    return jsonify({"name": tool_name})

@app.route('/api/tools/run/<name>', methods=['POST'])
def run_tool_api(name):
    files_json = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-5:] == '.json']
    files_py = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-3:] == '.py']
    if f"{name}.json" not in files_json or f"{name}.py" not in files_py:
        return jsonify({"error": "Tool not found"}), 404

    # Run tool directly as an API endpoint
    args_json = request.get_json(silent=True)
    args_json = json.dumps(args_json if args_json else {}) 
    result = execute_tool(name, args_json)
    try:
        return json.loads(result)
    except:
        return result

@app.route('/api/tools/<name>', methods=['DELETE'])
def delete_tool(name):
    files_json = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-5:] == '.json']
    files_py = [f for f in os.listdir('tools') if os.path.isfile(os.path.join('tools', f)) and f[-3:] == '.py']
    if f"{name}.json" not in files_json or f"{name}.py" not in files_py:
        return jsonify({"error": "Tool not found"}), 404
    os.remove(f"tools/{name}.json")
    os.remove(f"tools/{name}.py")
    return jsonify({"status": "deleted"})


# --- PAGES API ---
PAGE_INSTRUCTION = "You are a senior front-end engineer. Write code mainly using HTML, CSS, and JS unless the user ask for specify library. Respond only the code block and not conversational text."

@app.route('/api/pages', methods=['GET'])
def get_pages():
    pages = []
    for page_file in glob("pages/*.html"):
        pages.append({"name": os.path.basename(page_file).replace(".html", "")})
    return jsonify(pages)

@app.route('/api/pages/<name>', methods=['GET'])
def get_page_content(name):
    try:
        files = [f for f in os.listdir('pages') if os.path.isfile(os.path.join('pages', f)) and f[-5:] == '.html']
        if f"{name}.html" not in files:
            return jsonify({"error": "Page not found"}), 404
        with open(f"pages/{name}.html", "r") as f:
            return jsonify({"name": name, "content": f.read()})
    except FileNotFoundError:
        return jsonify({"error": "Page not found"}), 404

@app.route('/api/pages', methods=['POST'])
def generate_page():
    prompt = request.json.get("prompt")
    existing_name = request.json.get("existing_name")
    
    client = get_client("code_ai")
    settings = get_settings()["code_ai"]
    
    messages = [{"role": "system", "content": PAGE_INSTRUCTION}]
    
    if existing_name:
        if not is_safe_filename(existing_name):
            return {"error":"invalid file name",},404

        files = [f for f in os.listdir('pages') if os.path.isfile(os.path.join('pages', f)) and f[-5:] == '.html']
        if f"{existing_name}.html" not in files:
            return jsonify({"error": "Page not found"}), 404

        with open(f"pages/{existing_name}.html", "r") as f:
            old_code = f.read()
        messages.append({"role": "user", "content": f"Current Code:\n```html\n{old_code}\n```\n\nEdit instruction: {prompt}"})
    else:
        existing_name = f"page_{uuid.uuid4().hex[:8]}"
        messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=settings["model_name"],
        messages=messages,
        temperature=settings["temperature"]
    )
    
    reply = response.choices[0].message.content
    
    # Extract HTML block if wrapper exists, otherwise use raw reply
    html_match = re.search(r'```(?:html|HTML)\n(.*?)\n```', reply, re.DOTALL)
    html_content = html_match.group(1) if html_match else reply.replace('```', '')

    with open(f"pages/{existing_name}.html", "w") as f:
        f.write(html_content.strip())

    return jsonify({"name": existing_name, "endpoint": f"/pages/web/{existing_name}"})

@app.route('/api/pages/<name>/rename', methods=['PUT'])
def rename_page(name):
    new_name = request.json.get("new_name")
    
    if not new_name:
        return jsonify({"error": "new_name is required"}), 404
        
    if not is_safe_filename(new_name):
        return jsonify({"error": "invalid file name"}), 404
        
    old_path = f"pages/{name}.html"
    new_path = f"pages/{new_name}.html"
    
    # Check if the file exists[cite: 1]
    files = [f for f in os.listdir('pages') if os.path.isfile(os.path.join('pages', f)) and f[-5:] == '.html']
    if f"{name}.html" not in files:
        return jsonify({"error": "Page not found"}), 404
        
    # Prevent overwriting an existing page
    if f"{new_name}.html" in files:
        return jsonify({"error": "A page with the new name already exists"}), 409
        
    # Rename the file
    os.rename(old_path, new_path)
    
    return jsonify({
        "old_name": name,
        "new_name": new_name,
        "endpoint": f"/pages/web/{new_name}"
    })

@app.route('/pages/web/<name>', methods=['GET'])
def serve_page(name):
    # This serves the actual HTML page to a browser
    path = f"pages/{name}.html"
    files = [f for f in os.listdir('pages') if os.path.isfile(os.path.join('pages', f)) and f[-5:] == '.html']
    if f"{name}.html" not in files:
        return jsonify({"error": "Page not found"}), 404
    return send_file(path)

@app.route('/api/pages/<name>', methods=['DELETE'])
def delete_page(name):
    path = f"pages/{name}.html"
    files = [f for f in os.listdir('pages') if os.path.isfile(os.path.join('pages', f)) and f[-5:] == '.html']
    if f"{name}.html" not in files:
        return jsonify({"error": "Page not found"}), 404
    os.remove(path)
    return jsonify({"status": "deleted"})



# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled Exception: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

# Frontend Vue
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_vue(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host="127.0.0.1",port=5000)