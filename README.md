COMPANYBOT
AN AI CHATBOT FOR REALTIME COMPANY INFORMATION RETRIEVAL AND USER ENGAGEMENT
This project involves creating an AIpowered chatbot utilising the GPT4o model to generate responses, manage and query document data, and provide a user frontend for interaction. It evaluates models, manages contextual information, and utilizes language processing capabilities effectively.
 Project Features
1.	Chatbot Interaction:  Users can interact with a chatbot powered by GPT4o model via a Gradio interface.
2.	Document Database Management:  Efficiently handles the loading, splitting, and updating of document data using the Chroma vector store.
3.	Model Evaluation:  Compares different language models' responses based on Accuracy, semantic similarity and response time.

  Input and Output Data
   Input Data
•	Documents: PDFs located in a directory specified by `DATA_PATH`, used for developing contextual information for the chatbot.
•	Test Data:  An Excel file containing questions and their expected answers. The file path must be set using the environment variable `TEST_DATA`.
   Output Data
•	Evaluation Results:  Outputs include model responses, similarity scores to expected answers, and response times, saved in an Excel file defined by `METRIX`.
•	Visualizations:  Generates bar plots comparing model accuracies, response times, and semantic similarities.
Core Functions
Document Handling: 
•	`load_documents()`: Loads PDF documents from the specified directory.
•	`split_documents(documents)`: Splits large documents into smaller, manageable chunks.
•	`add_to_chroma(chunks)`: Adds new document chunks to the Chroma vector store after calculating unique IDs.
•	`check_clear_database()`: Resets the document database if the `reset` flag is specified.
  Chatbot Functions: 
•	`chatbot_chat(query_text, history)`: Processes user input, checks scope, queries relevant context, and generates responses using GPT models.
•	`chatbot_chat_test(query_text, model)`: A test function for generating responses from various models during development.

 Model Functions: 
•	`generate_gpt_response(model, prompt)`: Generates responses from OpenAI models.
•	`generate_t5_response(query_text, context_text)`: Uses a T5 model to produce responses based on input query and context.
 Parameters and Hyperparameters
•	Cosine Similarity Threshold:  Set to 0.7 for calculating response accuracy.
•	T5 Token Length:  Configured to generate responses with a maximum of 50 tokens.
•	Database Reset Flag:  Uses `reset` to clear existing document data.
Description of What the Code Does
This project consists of various components that work together to create an AI-driven chatbot system with document management capabilities. Here is a breakdown of each major functionality provided by the code:
1.  Chatbot Functionality: 
•	The chatbot interacts with users through a Gradio based frontend.
•	It uses GPT4o language models to generate responses.
•	Implements checks for greeting messages and out of scope questions (e.g., requests for personal information), ensuring responses remain relevant and appropriate.
2.  Document Database Management: 
•	Loads documents (PDFs) from a specified directory and splits them into chunks for better processing.
•	Utilizes Chroma, a vector store, to index and manage these document chunks, enabling efficient similarity searches.
•	Provides functionality for resetting the database, clearing all data when needed, using a command line flag.
3.  Model Evaluation: 
•	Compares the performance of different models by calculating similarity scores between the generated and expected responses.
•	Measures the response time of each model, providing insights into their efficiency.
•	Generates evaluation metrics, which are saved in an Excel file, and visualizes these metrics through bar charts.
4.  Response Generation: 
•	Uses OpenAI's GPT models to generate responses based on a context and query provided to the models.
•	For T5 models (used for result analysis), construct responses by encoding the query and context into a suitable format for the model to generate text.

This structured approach allows for sophisticated language processing capabilities, enabling the chatbot to provide accurate and contextually relevant responses while also maintaining an efficient document retrieval system.

 Challenges and Considerations
•	Managing pdf files while maintaining performance and efficiency.
•	Ensuring different models adhere to the predefined context limit and response quality.
•	Integrating multiple tools and libraries while ensuring compatibility and seamless operation.
  Getting Started
1.  Environment Setup:  Use a `.env` file to define `API_KEY` and path variables (`TEST_DATA`, `METRIX`, `CHROMA_PATH`, `DATA_PATH`).
2.  Installation:  Install necessary packages using:
   ```bash
   pip install r requirements.txt
   ```
3.  Database Management:  Use the `reset` flag to clear the database when required.
4.  Running the Application:  Execute the script to start the chatbot server and manage document indexing.
This project demonstrates an advanced approach to developing a versatile AI chatbot system using modern language models, document handling strategies, and an interactive interface, providing a robust foundation for further enhancements and real-world applications.

