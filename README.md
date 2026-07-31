### Warning

I was using AI while creating this project, it may contain (should not) security issues!

## Lets start!

1. Create a venv using Python 3.13 and install dependencies:

    ```bash
    python3.13 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    ```

2. Create `.env`:

    Then fill in the required values:

    ```env
    API_ID=123456
    API_HASH=telegram_api_hash
    GOOGLE_API_KEY=google_ai_key
    BOT_USERNAME=your_telegram_username
    RANDOM_REPLY_CHANCE=0.3
    SYSTEM_PROMPT_FILE=prompt.txt
    ```
    
    Project itself already contains `.env_example`.

3. Make your own system prompt in `prompt.txt` and run:

   ```bash
   .venv/bin/python main.py
   ```

## Authorization:

1. Get your own telegram token [telegram](https://my.telegram.org/auth?to=apps)

2. Then you run app, scan a QR-code in terminal

3. Follow any additional authorization instructions shown in the terminal.

4. Enjoy using it!

## Models support:

At this moment, programm is focused on Gemini (because it gives you 500 requests per day).
The model provider layer is designed so that support for other APIs can be added in the future.


