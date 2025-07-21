// This script will run once the DOM is fully loaded
window.addEventListener('DOMContentLoaded', (event) => {

    // Function to set up the speech recognition functionality
    const setupSpeechRecognition = () => {
        // Get references to the HTML elements we need to interact with
        const micButton = document.getElementById('mic-button');
        const chatInput = document.getElementById('chat-input-fixed');
        const fixedInputContainer = document.getElementById('fixed-input-container');
        const sendButton = document.getElementById('send-button-fixed');

        // If any of the required elements don't exist, stop.
        if (!micButton || !chatInput || !fixedInputContainer) {
            console.error("One or more required elements for voice dictation are not found.");
            return;
        }

        // Check if the browser supports the Web Speech API
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.error("Speech Recognition not supported by this browser.");
            micButton.style.display = 'none'; // Hide the mic button if the feature is not supported
            return;
        }

        // Create a new SpeechRecognition instance
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = true;

        let isListening = false;
        let final_transcript = '';

        // Event handler for when recognition starts
        recognition.onstart = () => {
            isListening = true;
            final_transcript = chatInput.value ? chatInput.value + ' ' : '';
            fixedInputContainer.classList.add('hearing');
            micButton.classList.add('listening');
        };

        // Event handler for any errors
        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
        };

        // Event handler for when recognition ends
        recognition.onend = () => {
            isListening = false;
            fixedInputContainer.classList.remove('hearing');
            micButton.classList.remove('listening');
            
            const finalValue = final_transcript.trim();

            // This is an attempt to force React to acknowledge the programmatic change.
            if (chatInput._valueTracker) {
              chatInput._valueTracker.setValue('');
            }

            // Use the native setter to update the DOM value
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
            nativeInputValueSetter.call(chatInput, finalValue);

            // Dispatch the 'input' event to trigger the onChange callback in Dash
            const inputEvent = new Event('input', { bubbles: true });
            chatInput.dispatchEvent(inputEvent);

            // Set focus to the send button
            sendButton.focus();
        };

        // Event handler for when a result is received
        recognition.onresult = (event) => {
            let interim_transcript = '';
            
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    final_transcript += event.results[i][0].transcript;
                } else {
                    interim_transcript += event.results[i][0].transcript;
                }
            }
            
            chatInput.value = (final_transcript + interim_transcript).trim();
        };

        // Add a click listener to the microphone button
        micButton.addEventListener('click', () => {
            if (isListening) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
    }

    // Since Dash loads content dynamically, we need to wait for the mic button to appear.
    const intervalId = setInterval(() => {
        const micButton = document.getElementById('mic-button');
        if (micButton) {
            clearInterval(intervalId);
            setupSpeechRecognition();
        }
    }, 500);
});