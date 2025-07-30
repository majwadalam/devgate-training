"""
Text analyzer tool for AI agents
"""

import re
from typing import Dict, List, Any
from collections import Counter

class TextAnalyzer:
    """Text analysis tool with various text processing capabilities"""
    
    def word_count(self, text: str) -> int:
        """Count number of words in text"""
        return len(text.split())
    
    def character_count(self, text: str, include_spaces: bool = True) -> int:
        """Count number of characters in text"""
        if include_spaces:
            return len(text)
        else:
            return len(text.replace(" ", ""))
    
    def sentence_count(self, text: str) -> int:
        """Count number of sentences in text"""
        # Simple sentence splitting based on punctuation
        sentences = re.split(r'[.!?]+', text.strip())
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)
    
    def paragraph_count(self, text: str) -> int:
        """Count number of paragraphs in text"""
        paragraphs = text.split('\n\n')
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return len(paragraphs)
    
    def word_frequency(self, text: str, top_n: int = 10) -> List[tuple]:
        """Get most frequent words in text"""
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Return top N most common words
        return word_counts.most_common(top_n)
    
    def extract_keywords(self, text: str, min_length: int = 4) -> List[str]:
        """Extract keywords (words longer than min_length)"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter by length and remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'since', 'without',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        keywords = [
            word for word in words 
            if len(word) >= min_length and word not in stop_words
        ]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for word in keywords:
            if word not in seen:
                unique_keywords.append(word)
                seen.add(word)
        
        return unique_keywords[:20]  # Return top 20 keywords
    
    def readability_score(self, text: str) -> Dict[str, float]:
        """Calculate basic readability metrics"""
        word_count = self.word_count(text)
        sentence_count = self.sentence_count(text)
        char_count = self.character_count(text, include_spaces=False)
        
        if sentence_count == 0 or word_count == 0:
            return {
                "avg_words_per_sentence": 0,
                "avg_chars_per_word": 0,
                "readability_score": 0
            }
        
        avg_words_per_sentence = word_count / sentence_count
        avg_chars_per_word = char_count / word_count
        
        # Simple readability score (lower is easier to read)
        # Based on Flesch Reading Ease concept (simplified)
        readability = (avg_words_per_sentence * 1.015) + (avg_chars_per_word * 84.6)
        
        return {
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_chars_per_word": round(avg_chars_per_word, 2),
            "readability_score": round(readability, 2)
        }
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        # Pattern to match integers and decimals
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        numbers = []
        for match in matches:
            try:
                if '.' in match:
                    numbers.append(float(match))
                else:
                    numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis using word lists"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'marvelous',
            'perfect', 'best', 'love', 'like', 'enjoy', 'happy', 'pleased',
            'satisfied', 'delighted', 'thrilled', 'excited', 'positive'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'sad', 'angry', 'frustrated', 'disappointed', 'upset',
            'annoyed', 'worried', 'concerned', 'problem', 'issue', 'wrong',
            'error', 'fail', 'failure', 'poor', 'worst', 'negative'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            confidence = 0.5
        elif positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / total_sentiment_words
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / total_sentiment_words
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Create a simple extractive summary"""
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring: prefer sentences with more words and keywords
        keywords = self.extract_keywords(text, min_length=3)
        keyword_set = set(keywords)
        
        sentence_scores = []
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            keyword_matches = len(sentence_words.intersection(keyword_set))
            word_count = len(sentence_words)
            
            # Score based on keyword matches and sentence length
            score = keyword_matches * 2 + word_count * 0.1
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'

# Tool functions for agent integration
def create_text_analyzer_tools() -> Dict[str, Any]:
    """Create text analyzer tool functions for agent use"""
    analyzer = TextAnalyzer()
    
    return {
        "word_count": {
            "function": analyzer.word_count,
            "description": "Count words in text",
            "parameters": ["text: str"]
        },
        "analyze_readability": {
            "function": analyzer.readability_score,
            "description": "Analyze text readability",
            "parameters": ["text: str"]
        },
        "extract_keywords": {
            "function": analyzer.extract_keywords,
            "description": "Extract keywords from text",
            "parameters": ["text: str", "min_length: int = 4"]
        },
        "sentiment_analysis": {
            "function": analyzer.sentiment_analysis,
            "description": "Analyze sentiment of text",
            "parameters": ["text: str"]
        },
        "summarize": {
            "function": analyzer.summarize_text,
            "description": "Create text summary",
            "parameters": ["text: str", "max_sentences: int = 3"]
        },
        "extract_numbers": {
            "function": analyzer.extract_numbers,
            "description": "Extract numbers from text",
            "parameters": ["text: str"]
        }
    }

if __name__ == "__main__":
    # Test the text analyzer
    analyzer = TextAnalyzer()
    
    sample_text = """
    Artificial intelligence is a fascinating field of computer science. 
    It involves creating machines that can think and learn like humans. 
    AI has many applications including natural language processing, 
    computer vision, and robotics. The future of AI looks very promising 
    with continued advances in machine learning and deep learning technologies.
    """
    
    print("Text Analysis Test:")
    print(f"Word count: {analyzer.word_count(sample_text)}")
    print(f"Sentence count: {analyzer.sentence_count(sample_text)}")
    print(f"Keywords: {analyzer.extract_keywords(sample_text)}")
    print(f"Sentiment: {analyzer.sentiment_analysis(sample_text)}")
    print(f"Readability: {analyzer.readability_score(sample_text)}")
    print(f"Summary: {analyzer.summarize_text(sample_text, 2)}") 