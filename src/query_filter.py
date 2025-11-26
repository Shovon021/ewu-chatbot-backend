"""
Query Filter Module
Validates if queries are academic/university-related before processing.
"""

import re
from typing import Tuple


class QueryFilter:
    """Filter to validate if queries are academic/university-related."""
    
    # Academic keywords that indicate relevant queries
    ACADEMIC_KEYWORDS = {
        # University-specific
        'ewu', 'east west university', 'university', 'campus',
        
        # Department and programs
        'cse', 'computer science', 'engineering', 'department',
        'undergraduate', 'graduate', 'bachelor', 'master', 'phd',
        'b.sc', 'm.sc', 'program', 'major', 'degree',
        
        # Courses and academics
        'course', 'class', 'subject', 'syllabus', 'curriculum',
        'credit', 'semester', 'prerequisite', 'lab', 'theory',
        
        # Faculty and staff
        'professor', 'faculty', 'teacher', 'instructor', 'staff',
        'chairperson', 'dean', 'lecturer', 'dr.', 'doctor',
        
        # Admissions and fees
        'admission', 'enroll', 'registration', 'tuition', 'fee',
        'scholarship', 'financial aid', 'application',
        
        # Facilities and resources
        'library', 'laboratory', 'facility', 'equipment',
        
        # Academic activities
        'research', 'project', 'thesis', 'publication',
        'exam', 'test', 'grade', 'gpa', 'cgpa',
        
        # Administrative
        'office', 'contact', 'email', 'phone', 'address',
        'schedule', 'timing', 'hours', 'location'
    }
    
    # Non-academic keywords that indicate irrelevant queries
    NON_ACADEMIC_KEYWORDS = {
        'weather', 'news', 'sports', 'movie', 'music',
        'recipe', 'cooking', 'restaurant', 'food',
        'game', 'gaming', 'entertainment',
        'shopping', 'buy', 'sell', 'price',
        'stock', 'market', 'cryptocurrency',
        'joke', 'funny', 'meme'
    }
    
    # Greeting patterns (allowed but handled separately)
    GREETING_PATTERNS = [
        r'\b(hi|hello|hey|greetings?|good\s+(morning|afternoon|evening|day))\b',
        r'\b(bye|goodbye|see\s+you|farewell)\b',
        r'\b(thanks?|thank\s+you|appreciate)\b'
    ]
    
    def __init__(self):
        """Initialize the query filter."""
        pass
    
    def is_greeting(self, query: str) -> bool:
        """
        Check if query is a greeting.
        
        Args:
            query: User query text
            
        Returns:
            True if query is a greeting, False otherwise
        """
        query_lower = query.lower().strip()
        
        for pattern in self.GREETING_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def is_academic_query(self, query: str) -> Tuple[bool, str]:
        """
        Determine if a query is academic/university-related.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (is_academic, reason)
        """
        query_lower = query.lower().strip()
        
        # Empty query
        if not query or len(query.strip()) < 3:
            return False, "Query too short"
        
        # Check for greetings (allowed)
        if self.is_greeting(query):
            return True, "greeting"
        
        # Check for non-academic keywords first
        for keyword in self.NON_ACADEMIC_KEYWORDS:
            if keyword in query_lower:
                return False, f"Non-academic keyword detected: '{keyword}'"
        
        # Check for academic keywords
        for keyword in self.ACADEMIC_KEYWORDS:
            if keyword in query_lower:
                return True, f"Academic keyword detected: '{keyword}'"
        
        # Check for course codes (CSE103, MAT101, etc.)
        if re.search(r'\b[A-Z]{2,4}\s*\d{3}\b', query):
            return True, "Course code detected"
        
        # If no clear indicators, be conservative but allow
        # (Let the LLM decide if it can answer from context)
        if len(query.split()) <= 3:
            # Short queries without keywords are likely too vague
            return False, "Query too vague - please be more specific about EWU/CSE topics"
        
        # Default: assume academic if reasonably long
        return True, "General academic query assumed"
    
    def get_refusal_message(self, query: str) -> str:
        """
        Get appropriate refusal message for non-academic queries.
        
        Args:
            query: User query text
            
        Returns:
            Polite refusal message
        """
        return (
            "I apologize, but I'm the EWU University Academic Assistant, and I can only help with questions related to:\n\n"
            "• East West University (EWU)\n"
            "• Computer Science & Engineering Department\n"
            "• Academic programs, courses, and faculty\n"
            "• Admissions, scholarships, and fees\n"
            "• University policies and administration\n\n"
            "Please ask me a question about EWU or the CSE department, and I'll be happy to help!"
        )


def is_academic_query(query: str) -> bool:
    """
    Helper function to check if query is academic.
    
    Args:
        query: User query text
        
    Returns:
        True if academic, False otherwise
    """
    filter = QueryFilter()
    is_academic, _ = filter.is_academic_query(query)
    return is_academic


if __name__ == "__main__":
    # Test the query filter
    print("Testing Query Filter...\n")
    
    filter = QueryFilter()
    
    test_queries = [
        "What is CSE366?",
        "Who is the department chairperson?",
        "What's the weather today?",
        "Tell me a joke",
        "What scholarships are available?",
        "Hello!",
        "How to cook pasta?",
    ]
    
    for query in test_queries:
        is_academic, reason = filter.is_academic_query(query)
        status = "✓ ACADEMIC" if is_academic else "✗ NON-ACADEMIC"
        print(f"{status}: '{query}'")
        print(f"  Reason: {reason}\n")
