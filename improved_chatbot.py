import os
import pandas as pd
import logging
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ImprovedChatbot:
    """A chatbot that uses RAG (Retrieval Augmented Generation) to answer questions
    about aesthetic medical treatments offered by Hautlabor.
    """
    
    def __init__(self, api_key, csv_file="enhanced_kb_chunks.csv"):
        """Initialize improved chatbot with memory and faster responses
        
        Args:
            api_key (str): OpenAI API key (should start with 'sk-')
            csv_file (str): Path to the knowledge base CSV file
        """
        # Validate and clean API key
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing ImprovedChatbot")
        
        # Clean and validate API key
        cleaned_api_key = self._clean_api_key(api_key)
        if not self._validate_api_key(cleaned_api_key):
            self.logger.error("Invalid API key format")
            raise ValueError("Invalid API key format. API key should start with 'sk-'")
            
        os.environ["OPENAI_API_KEY"] = cleaned_api_key

        # Use faster, cheaper model
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # Much faster than gpt-4
            temperature=0.1,
            max_tokens=500  # Limit response length for speed
        )

        self.embeddings = OpenAIEmbeddings()

        # Add conversation memory (fixes memory issue)
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Load and filter data
        try:
            self.load_and_filter_data(csv_file)
            self.build_vectorstore()
            self.create_chain()
            self.logger.info("Chatbot initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise

    def _clean_api_key(self, api_key):
        """Clean the API key by removing any prefix like 'Y' or extra characters"""
        if not api_key:
            return ""
            
        # If the key has a 'Y' prefix, remove it
        if api_key.startswith('Y') and len(api_key) > 1:
            api_key = api_key[1:]
            
        # Clean any whitespace
        return api_key.strip()
        
    def _validate_api_key(self, api_key):
        """Validate that the API key has the correct format (starts with sk-)"""
        if not api_key:
            return False
            
        # Basic validation - should start with sk-
        return api_key.startswith('sk-') and len(api_key) > 20
    
    def load_and_filter_data(self, csv_file):
        """Load and filter chunks to improve quality
        
        Args:
            csv_file (str): Path to the CSV file containing the knowledge base
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If the CSV file is missing required columns
        """
        self.logger.info(f"Loading data from {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            
            # Verify the CSV has the required columns
            required_columns = ['importance', 'has_treatments', 'type', 'text', 'source']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"CSV file missing required columns: {missing_columns}")
                raise ValueError(f"CSV file is missing required columns: {missing_columns}")
                
            self.logger.info(f"Loaded dataframe with {len(df)} rows")

            # Handle potentially empty dataframe
            if len(df) == 0:
                self.logger.warning("CSV file is empty. Using an empty dataset.")
                filtered_df = df.copy()
            else:
                # Filter out low-quality chunks
                try:
                    filtered_df = df[
                        (df['importance'] >= 15) |  # High importance chunks
                        (df['has_treatments'] == True) |  # Treatment-related content
                        (df['type'].isin(['treatment_page', 'homepage', 'summary']))  # Important page types
                    ].copy()

                    # Remove navigation/menu noise
                    filtered_df = filtered_df[
                        ~filtered_df['text'].str.contains('Schließen|Startseite|Über uns', na=False, case=False)
                    ]
                    
                    self.logger.info(f"Filtered to {len(filtered_df)} high-quality chunks")
                except Exception as e:
                    self.logger.warning(f"Error during filtering: {str(e)}. Using all data.")
                    filtered_df = df.copy()
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            # Create an empty DataFrame with the required columns
            filtered_df = pd.DataFrame(columns=['importance', 'has_treatments', 'type', 'text', 'source'])
            self.logger.warning("Created empty dataframe due to loading error")

        print(f"Filtered from {len(df)} to {len(filtered_df)} high-quality chunks")

        # Convert to documents
        self.documents = []
        
        # If we have no data after filtering, add a placeholder document
        if len(filtered_df) == 0:
            self.logger.warning("No data available after filtering. Adding placeholder document.")
            self.documents.append(Document(
                page_content="Hautlabor Dr. med. Lara Pfahl bietet verschiedene ästhetische Behandlungen an.",
                metadata={
                    'source': 'placeholder',
                    'type': 'homepage',
                    'importance': 20
                }
            ))
        else:
            for _, row in filtered_df.iterrows():
                try:
                    doc = Document(
                        page_content=row['text'],
                        metadata={
                            'source': row.get('source', 'unknown'),
                            'type': row.get('type', 'general'),
                            'importance': row.get('importance', 0)
                        }
                    )
                    self.documents.append(doc)
                except Exception as e:
                    self.logger.warning(f"Error processing row: {str(e)}")

    def build_vectorstore(self):
        """Build optimized vector store"""
        if not self.documents:
            self.logger.warning("No documents to build vectorstore. Creating placeholder document.")
            placeholder = Document(
                page_content="Hautlabor Dr. med. Lara Pfahl bietet verschiedene ästhetische Behandlungen an.",
                metadata={
                    'source': 'placeholder',
                    'type': 'homepage',
                    'importance': 20
                }
            )
            self.documents = [placeholder]
            
        try:
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            self.logger.info(f"Built vectorstore with {len(self.documents)} documents")
            print(f"Built vectorstore with {len(self.documents)} documents")
        except Exception as e:
            self.logger.error(f"Error building vectorstore: {str(e)}")
            raise

    def create_chain(self):
        """Create conversational chain with memory"""

        # Improved prompt template
        custom_prompt = PromptTemplate(
            template="""Du bist der AI-Assistent von Dr. med. Lara Pfahl für das Hautlabor Oldenburg.

Verwende den folgenden Kontext und die Gesprächshistorie, um die Frage zu beantworten:

Kontext: {context}

Gesprächshistorie: {chat_history}

Aktuelle Frage: {question}

Anweisungen:
- Gib präzise, hilfreiche Antworten auf Deutsch
- Bei Behandlungsfragen: nenne Kosten, Dauer und Ablauf wenn verfügbar
- Verweise auf Beratungstermine: +49 (0) 157 834 488 90
- Sei freundlich und professionell
- Halte Antworten fokussiert (max. 3-4 Sätze)

Antwort:""",
            input_variables=["context", "chat_history", "question"]
        )

        # Optimized retriever (faster)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",  # Faster than MMR
            search_kwargs={"k": 5}  # Fewer chunks = faster
        )

        # Create conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            verbose=False
        )

        print("✅ Conversational chain created with memory")

    def chat(self, question):
        """Chat with memory and context
        
        Args:
            question (str): User's question
            
        Returns:
            str: Chatbot's answer to the question
        """
        if not question or not isinstance(question, str):
            self.logger.warning(f"Invalid question: {question}")
            return "Bitte stellen Sie eine Frage."
            
        self.logger.info(f"Processing question: {question[:50]}...")
        
        try:
            # Get response
            result = self.chain.invoke({"question": question})

            # Format response
            answer = result.get("answer", "")
            if not answer:
                self.logger.warning("Empty answer received from model")
                answer = "Entschuldigung, ich konnte keine Antwort generieren."
                
            sources = result.get("source_documents", [])

            print(f"\n🤖 Antwort: {answer}")
            self.logger.info(f"Generated answer: {answer[:50]}...")

            if sources:
                source_info = []
                print(f"\n📚 Quellen:")
                for i, doc in enumerate(sources[:2], 1):  # Show top 2 sources
                    source = doc.metadata.get('source', 'Unbekannt')
                    print(f"  {i}. {source}")
                    source_info.append(source)
                self.logger.info(f"Sources used: {source_info}")

            return answer

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            error_msg = f"Entschuldigung, es gab einen Fehler: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def reset_memory(self):
        """Reset conversation memory"""
        self.memory.clear()
        print("🧠 Gesprächshistorie zurückgesetzt")

