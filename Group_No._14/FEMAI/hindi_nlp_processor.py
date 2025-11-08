"""
Hindi NLP Processor
Extracts structured data from Hindi/English voice input
"""
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class HindiNLPProcessor:
    def __init__(self):
        # Hindi-English keyword mappings
        self.keywords = {
            'age': {
                'hindi': ['उम्र', 'आयु', 'साल', 'वर्ष', 'उमर'],
                'english': ['age', 'years', 'year', 'old'],
                'patterns': [r'(\d+)\s*(?:साल|वर्ष|years?|old)?']
            },
            'bmi': {
                'hindi': ['बीएमआई', 'वजन', 'भार'],
                'english': ['bmi', 'body mass index', 'weight'],
                'patterns': [r'(\d+\.?\d*)\s*(?:bmi|बीएमआई)?']
            }
        }

        # Yes/No responses
        self.affirmative = {
            'hindi': ['हां', 'हाँ', 'जी', 'जी हां', 'बिल्कुल', 'सही', 'ठीक'],
            'english': ['yes', 'yeah', 'yep', 'correct', 'right', 'true', 'affirmative']
        }

        self.negative = {
            'hindi': ['नहीं', 'नही', 'ना', 'गलत'],
            'english': ['no', 'nope', 'nah', 'not', 'wrong', 'negative']
        }

        # Number words (Hindi and English)
        self.number_words = {
            # Hindi
            'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पांच': 5,
            'छह': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
            'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14, 'पंद्रह': 15,
            'सोलह': 16, 'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19, 'बीस': 20,
            'पच्चीस': 25, 'तीस': 30, 'पैंतीस': 35, 'चालीस': 40, 'पैंतालीस': 45,
            # English
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'twenty-five': 25, 'thirty': 30, 'thirty-five': 35, 'forty': 40, 'forty-five': 45,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }

    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract number from text (supports Hindi and English)

        Returns:
            Number (int or float) or None
        """
        text = text.lower().strip()

        # First try number words
        for word, num in self.number_words.items():
            if word in text:
                return num

        # Try digit extraction
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            try:
                num = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                return num
            except ValueError:
                pass

        return None

    def detect_yes_no(self, text: str) -> Optional[int]:
        """
        Detect yes/no from Hindi/English text

        Returns:
            1 for yes, 0 for no, None if unclear
        """
        text = text.lower().strip()

        # Check affirmative
        for word in self.affirmative['hindi'] + self.affirmative['english']:
            if word in text:
                return 1

        # Check negative
        for word in self.negative['hindi'] + self.negative['english']:
            if word in text:
                return 0

        return None

    def process_answer(self, text: str, question_type: str, question_id: str) -> Dict[str, Any]:
        """
        Process voice answer based on question type

        Args:
            text: Transcribed text
            question_type: 'number', 'yes_no', 'select'
            question_id: ID of the question (e.g., 'age', 'bmi')

        Returns:
            Dictionary with extracted value and confidence
        """
        result = {
            'success': False,
            'value': None,
            'confidence': 'low',
            'raw_text': text
        }

        if not text or not text.strip():
            return result

        try:
            if question_type == 'number':
                value = self.extract_number(text)
                if value is not None:
                    # Validate range based on question
                    if self._validate_number_range(question_id, value):
                        result['success'] = True
                        result['value'] = value
                        result['confidence'] = 'high'
                    else:
                        result['confidence'] = 'medium'
                        result['value'] = value
                        result['error'] = 'Value out of expected range'

            elif question_type == 'yes_no':
                value = self.detect_yes_no(text)
                if value is not None:
                    result['success'] = True
                    result['value'] = value
                    result['confidence'] = 'high'

            elif question_type == 'select':
                # For dropdown selections
                value = self.extract_number(text)
                if value is not None:
                    result['success'] = True
                    result['value'] = int(value)
                    result['confidence'] = 'medium'

        except Exception as e:
            logger.error(f"Error processing answer: {e}")
            result['error'] = str(e)

        return result

    def _validate_number_range(self, question_id: str, value: float) -> bool:
        """Validate if number is in expected range for the question"""
        ranges = {
            'age': (10, 100),
            'bmi': (10, 60),
            'family': (0, 10),
            'miscarriage': (0, 20)
        }

        if question_id in ranges:
            min_val, max_val = ranges[question_id]
            return min_val <= value <= max_val

        return True  # No validation for unknown questions

    def generate_confirmation(self, question_id: str, value: Any, language: str = 'hindi') -> str:
        """
        Generate confirmation message

        Args:
            question_id: Question identifier
            value: Extracted value
            language: 'hindi' or 'english'

        Returns:
            Confirmation message
        """
        if language == 'hindi':
            confirmations = {
                'age': f"आपकी उम्र {value} साल है?",
                'bmi': f"आपका BMI {value} है?",
                'irregular': f"अनियमित माहवारी - {'हां' if value == 1 else 'नहीं'}?",
                'weightgain': f"वजन बढ़ना - {'हां' if value == 1 else 'नहीं'}?",
                'hairgrowth': f"अतिरिक्त बाल - {'हां' if value == 1 else 'नहीं'}?",
                'acne': f"मुहांसे - {'हां' if value == 1 else 'नहीं'}?",
                'hairloss': f"बाल गिरना - {'हां' if value == 1 else 'नहीं'}?",
                'family': f"परिवार में {value} सदस्य हैं?",
                'pain': f"दर्द - {'हां' if value == 1 else 'नहीं'}?",
                'infertility': f"गर्भधारण में कठिनाई - {'हां' if value == 1 else 'नहीं'}?",
                'miscarriage': f"गर्भपात {value} बार हुए हैं?"
            }
        else:
            confirmations = {
                'age': f"Your age is {value} years?",
                'bmi': f"Your BMI is {value}?",
                'irregular': f"Irregular periods - {'Yes' if value == 1 else 'No'}?",
                'weightgain': f"Weight gain - {'Yes' if value == 1 else 'No'}?",
                'hairgrowth': f"Excess hair growth - {'Yes' if value == 1 else 'No'}?",
                'acne': f"Acne - {'Yes' if value == 1 else 'No'}?",
                'hairloss': f"Hair loss - {'Yes' if value == 1 else 'No'}?",
                'family': f"{value} family members have PCOD/PCOS?",
                'pain': f"Pain - {'Yes' if value == 1 else 'No'}?",
                'infertility': f"Difficulty conceiving - {'Yes' if value == 1 else 'No'}?",
                'miscarriage': f"{value} miscarriages?"
            }

        return confirmations.get(question_id, f"Value: {value}?")