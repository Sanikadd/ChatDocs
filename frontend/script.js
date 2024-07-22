function formatResponse(text) {
    // Replace bold markers with <strong>
    let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
        .replace(/_(.*?)_/g, '<em>$1</em>');                // Italics

    // Handle list items
    let lines = formattedText.split('\n');
    let inList = false;
    let result = '';

    lines.forEach(line => {
        line = line.trim(); // Remove extra spaces
        if (line.startsWith('* ')) {
            if (!inList) {
                result += '<ul>';
                inList = true;
            }
            result += '<li>' + line.substring(2).trim() + '</li>';
        } else {
            if (inList) {
                result += '</ul>';
                inList = false;
            }
            result += '<p>' + line + '</p>';
        }
    });

    if (inList) {
        result += '</ul>';
    }

    return result;
}


function appendMessage(message, type) {
    console.log('Appending message:', message);  // Debugging statement
    const chatBox = document.querySelector('.chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}-message`;

    const formattedMessage = formatResponse(message);
    console.log('Formatted message:', formattedMessage);  // Debugging statement
    messageDiv.innerHTML = formattedMessage;  // Directly set the innerHTML

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll to the bottom
}



function uploadFiles() {
    const formData = new FormData(document.getElementById('uploadForm'));
    filesData = formData;  // Store files data for later use
    fetch('/submit_files', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            appendMessage('Files uploaded successfully', 'bot');
        } else if (data.error) {
            appendMessage(`Error: ${data.error}`, 'bot');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage(`Error: ${error}`, 'bot');
    });
}

function summarizeFiles() {
    appendMessage('Summarize', 'user');  // Add user message to chatbox
    fetch('/summarize', {
        method: 'POST',
        body: filesData
    })
    .then(response => response.json())
    .then(data => {
        if (data.summary) {
            appendMessage(`Summary: ${data.summary}`, 'bot');  // Add bot response to chatbox
        } else if (data.error) {
            appendMessage(`Error: ${data.error}`, 'bot');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage(`Error: ${error}`, 'bot');
    });
}

function askQuestion() {
    const question = document.getElementById('question').value;
    appendMessage(question, 'user');  // Add user message to chatbox
    fetch('/ask_question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'question': question })
    })
    .then(response => response.json())
    .then(data => {
        if (data.answer) {
            appendMessage(data.answer, 'bot');  // Add bot response to chatbox
        } else if (data.error) {
            appendMessage(`Error: ${data.error}`, 'bot');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage(`Error: ${error}`, 'bot');
    });
}
