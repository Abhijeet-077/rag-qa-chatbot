"""
RAG System Implementation with ROSE Framework
Implements the core RAG functionality with recursive prompting
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Structure for retrieval results"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str

@dataclass
class RAGResponse:
    """Structure for RAG system responses"""
    answer: str
    confidence: float
    sources: List[str]
    context_used: List[RetrievalResult]
    needs_clarification: bool = False
    clarification_questions: List[str] = None

class ROSEPromptEngine:
    """
    ROSE Framework Implementation for RAG System
    Role, Objective, Style, Execution
    """
    
    def __init__(self, system_role: str = "domain-aware business assistant"):
        self.system_role = system_role
        self.base_prompt = self._build_base_prompt()
        self.recursive_prompt = self._build_recursive_prompt()
    
    def _build_base_prompt(self) -> str:
        """Build the base system prompt using ROSE framework"""
        return f"""
**ROLE**: You are a {self.system_role} with expertise in analyzing business documents and providing accurate, contextual answers.

**OBJECTIVE**: 
- Provide precise, business-appropriate answers based on retrieved context
- Reduce manual workload by answering repetitive questions
- Boost customer satisfaction through accurate information delivery
- Improve knowledge accessibility across the organization

**STYLE**: 
- Professional and concise communication
- Structure responses with clear bullet points when appropriate
- Use business-formal tone
- Cite sources when referencing specific information
- Acknowledge limitations when context is insufficient

**EXECUTION**:
1. Analyze the user's question for intent and complexity
2. Evaluate the retrieved context for relevance and completeness
3. Synthesize information from multiple sources when necessary
4. Generate responses that directly address the user's needs
5. Request clarification when the question is ambiguous
6. Provide confidence indicators for your responses

Remember: Only use information from the provided context. If the context doesn't contain sufficient information to answer the question, acknowledge this limitation and suggest how the user might get the information they need.
"""
    
    def _build_recursive_prompt(self) -> str:
        """Build recursive clarification prompt"""
        return """
Before generating your final response, recursively evaluate:

1. **Clarity Check**: Is the user's question specific enough to provide a useful answer?
2. **Context Sufficiency**: Does the retrieved context contain enough relevant information?
3. **Relevance Assessment**: How well does the retrieved content match the question intent?
4. **Confidence Evaluation**: What is your confidence level in the answer (0-1 scale)?

If confidence < 0.7, consider:
- Requesting clarification from the user
- Suggesting alternative questions
- Providing partial answers with caveats

Generate follow-up questions if needed to improve answer quality.
"""

class RAGSystem:
    """Main RAG System implementing retrieval-augmented generation"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(config.pinecone_index_name)
        self.prompt_engine = ROSEPromptEngine(config.system_role)
        self.tokenizer = tiktoken.encoding_for_model(config.openai_model)
        
        logger.info("RAG System initialized successfully")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI's embedding model"""
        try:
            response = self.client.embeddings.create(
                model=config.embedding_model,
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _retrieve_context(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Retrieve relevant context from Pinecone vector database"""
        if top_k is None:
            top_k = config.top_k_results
        
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process results
            retrieved_contexts = []
            for match in search_results['matches']:
                result = RetrievalResult(
                    content=match.get('metadata', {}).get('text', ''),
                    score=match.get('score', 0.0),
                    metadata=match.get('metadata', {}),
                    source=match.get('metadata', {}).get('source', 'Unknown')
                )
                retrieved_contexts.append(result)
            
            logger.info(f"Retrieved {len(retrieved_contexts)} contexts for query")
            return retrieved_contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _build_context_string(self, contexts: List[RetrievalResult]) -> str:
        """Build context string from retrieved results"""
        if not contexts:
            return ""
        
        context_parts = []
        total_tokens = 0
        
        for i, context in enumerate(contexts):
            if context.score < config.confidence_threshold:
                continue
            
            context_text = f"[Source {i+1}: {context.source}]\n{context.content}\n"
            context_tokens = self._count_tokens(context_text)
            
            if total_tokens + context_tokens > config.max_context_length:
                break
            
            context_parts.append(context_text)
            total_tokens += context_tokens
        
        return "\n---\n".join(context_parts)
    
    def _evaluate_response_confidence(self, query: str, contexts: List[RetrievalResult]) -> float:
        """Evaluate confidence in the response based on context quality"""
        if not contexts:
            return 0.0
        
        # Calculate average relevance score
        avg_score = sum(c.score for c in contexts) / len(contexts)
        
        # Adjust for number of relevant contexts
        context_factor = min(len(contexts) / config.top_k_results, 1.0)
        
        # Penalize if no high-confidence matches
        high_conf_matches = sum(1 for c in contexts if c.score >= config.confidence_threshold)
        confidence_factor = high_conf_matches / len(contexts) if contexts else 0
        
        return avg_score * context_factor * confidence_factor
    
    def _generate_clarification_questions(self, query: str, contexts: List[RetrievalResult]) -> List[str]:
        """Generate clarification questions using recursive prompting"""
        clarification_prompt = f"""
Given the user query: "{query}"
And the available context quality (average relevance: {sum(c.score for c in contexts)/len(contexts) if contexts else 0:.2f})

Generate 2-3 clarification questions that would help provide a better answer:
1. Focus on ambiguous terms or concepts
2. Ask for specific use cases or scenarios
3. Clarify the level of detail needed

Format as a JSON array of strings.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates clarification questions."},
                    {"role": "user", "content": clarification_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = json.loads(questions_text)
            return questions if isinstance(questions, list) else [questions_text]
            
        except Exception as e:
            logger.error(f"Error generating clarification questions: {e}")
            return ["Could you please provide more specific details about your question?"]
    
    def query(self, user_query: str, conversation_history: List[Dict] = None) -> RAGResponse:
        """
        Main query method implementing the full RAG pipeline
        """
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: Retrieve relevant context
        contexts = self._retrieve_context(user_query)
        
        # Step 2: Evaluate confidence
        confidence = self._evaluate_response_confidence(user_query, contexts)
        
        # Step 3: Check if clarification is needed
        needs_clarification = confidence < config.confidence_threshold
        
        if needs_clarification:
            clarification_questions = self._generate_clarification_questions(user_query, contexts)
            return RAGResponse(
                answer="I need more information to provide an accurate answer.",
                confidence=confidence,
                sources=[],
                context_used=contexts,
                needs_clarification=True,
                clarification_questions=clarification_questions
            )
        
        # Step 4: Build context string
        context_string = self._build_context_string(contexts)
        
        # Step 5: Generate response
        messages = [
            {"role": "system", "content": self.prompt_engine.base_prompt},
            {"role": "system", "content": self.prompt_engine.recursive_prompt},
            {"role": "system", "content": f"Context Information:\n{context_string}"},
            {"role": "user", "content": user_query}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            sources = list(set(c.source for c in contexts if c.score >= config.confidence_threshold))
            
            return RAGResponse(
                answer=answer,
                confidence=confidence,
                sources=sources,
                context_used=contexts,
                needs_clarification=False
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your request. Please try again.",
                confidence=0.0,
                sources=[],
                context_used=contexts,
                needs_clarification=True,
                clarification_questions=["Could you please rephrase your question?"]
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        try:
            # Test OpenAI connection
            test_embedding = self._get_embedding("test")
            openai_status = "healthy" if test_embedding else "unhealthy"
            
            # Test Pinecone connection
            index_stats = self.index.describe_index_stats()
            pinecone_status = "healthy" if index_stats else "unhealthy"
            
            return {
                "openai_status": openai_status,
                "pinecone_status": pinecone_status,
                "index_stats": index_stats,
                "configuration": {
                    "model": config.openai_model,
                    "embedding_model": config.embedding_model,
                    "index_name": config.pinecone_index_name
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "openai_status": "unhealthy",
                "pinecone_status": "unhealthy",
                "error": str(e)
            }
