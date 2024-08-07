# AI-Powered Electronics Store Chatbot

## Description

This project is an advanced chatbot system designed for an electronics store, specializing in laptops and phones. It utilizes artificial intelligence to provide detailed product information, comparisons, and personalized recommendations to users. The system is built with a Flask backend, MongoDB for data storage, and integrates OpenAI's GPT-4 for natural language processing.

## Features

- AI-powered product queries and recommendations
- User authentication (login/signup)
- Chat history tracking
- Dynamic product inventory management
- Predefined question suggestions
- Multi-session support

## Technologies Used

- Backend: Python, Flask
- Database: MongoDB
- AI Model: OpenAI GPT-4
- Data Processing: Pandas
- API: RESTful API with JSON
- Security: CORS (Cross-Origin Resource Sharing)
- Others: UUID for session management, Logging for debugging

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-electronics-store-chatbot.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your MongoDB database and update the connection string in `constants.py`.

4. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

5. Run the application:
   ```
   python app.py
   ```

## Usage

1. Start the Flask server by running `app.py`.
2. Use the provided API endpoints to interact with the chatbot:
   - `/chatbot`: POST request to ask questions to the chatbot
   - `/login`: POST request for user login
   - `/signup`: POST request for user registration
   - `/history`: POST request to retrieve chat history
   - `/update_quantity`: POST request to update product quantities

## API Endpoints

- `POST /chatbot`: Send user queries to the chatbot
- `POST /login`: Authenticate users
- `POST /signup`: Register new users
- `POST /history`: Retrieve chat history for a user
- `POST /update_quantity`: Update product quantities in inventory

## Configuration

- Update `constants.py` with your specific database details and predefined questions.
- Modify the `promptttt` variable in `app.py` to customize the chatbot's behavior and responses.

## Data Management

The project uses a CSV file (`merged_store_db.csv`) as the primary data source for product information. This file is read and processed using Pandas for efficient data handling.

## Logging

The application uses Python's logging module for debugging and tracking. Logs are written to `app.log`.

## Contributing

Contributions to improve the chatbot or extend its functionality are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

[MIT License](LICENSE)

## Contact

Project Link: https://github.com/your-username/ai-electronics-store-chatbot
