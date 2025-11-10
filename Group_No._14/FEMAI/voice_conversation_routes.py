"""
Flask Routes for Voice Conversation Chatbot
Handles PCOD and PCOS voice assessments
"""
import os
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from voice_handler import VoiceHandler, AudioRecorder
from hindi_nlp_processor import HindiNLPProcessor
from voice_config import VOICE_UPLOAD_FOLDER

logger = logging.getLogger(__name__)

voice_conversation_bp = Blueprint('voice_conversation', __name__)

# Initialize handlers
transcriber = VoiceHandler(language='hindi')
nlp_processor = HindiNLPProcessor()
audio_validator = AudioRecorder()

# PCOD Questions
PCOD_QUESTIONS = [
    {
        'id': 'age',
        'question_hindi': 'आपकी उम्र क्या है?',
        'question_english': 'What is your age?',
        'hint_hindi': 'उदाहरण: "मेरी उम्र पच्चीस साल है" या "25"',
        'hint_english': 'Example: "Twenty five years" or "25"',
        'type': 'number',
        'validation': {'min': 10, 'max': 100}
    },
    {
        'id': 'bmi',
        'question_hindi': 'आपका BMI क्या है?',
        'question_english': 'What is your BMI?',
        'hint_hindi': 'उदाहरण: "बाईस" या "22"',
        'hint_english': 'Example: "Twenty two" or "22"',
        'type': 'number',
        'validation': {'min': 10, 'max': 60}
    },
    {
        'id': 'irregular',
        'question_hindi': 'क्या आपको अनियमित माहवारी की समस्या है?',
        'question_english': 'Do you have irregular periods?',
        'hint_hindi': '"हां" या "नहीं"',
        'hint_english': '"Yes" or "No"',
        'type': 'yes_no'
    },
    {
        'id': 'weightgain',
        'question_hindi': 'क्या आपका वजन बढ़ रहा है?',
        'question_english': 'Are you experiencing weight gain?',
        'hint_hindi': '"हां" या "नहीं"',
        'hint_english': '"Yes" or "No"',
        'type': 'yes_no'
    },
    {
        'id': 'hairgrowth',
        'question_hindi': 'क्या आपको चेहरे या शरीर पर अतिरिक्त बाल हैं?',
        'question_english': 'Do you have excess hair growth on face or body?',
        'hint_hindi': '"हां" या "नहीं"',
        'hint_english': '"Yes" or "No"',
        'type': 'yes_no'
    },
    {
        'id': 'acne',
        'question_hindi': 'क्या आपको मुहांसे या तैलीय त्वचा की समस्या है?',
        'question_english': 'Do you have acne or oily skin problems?',
        'hint_hindi': '"हां" या "नहीं"',
        'hint_english': '"Yes" or "No"',
        'type': 'yes_no'
    },
    {
        'id': 'hairloss',
        'question_hindi': 'क्या आपके बाल झड़ रहे हैं?',
        'question_english': 'Are you experiencing hair loss?',
        'hint_hindi': '"हां" या "नहीं"',
        'hint_english': '"Yes" or "No"',
        'type': 'yes_no'
    },
    {
        'id': 'family',
        'question_hindi': 'आपके परिवार में कितने सदस्यों को PCOD है?',
        'question_english': 'How many family members have PCOD?',
        'hint_hindi': 'उदाहरण: "दो" या "0"',
        'hint_english': 'Example: "Two" or "0"',
        'type': 'number',
        'validation': {'min': 0, 'max': 10}
    },
    {
        'id': 'pain',
        'question_hindi': 'क्या आपको पेट के निचले हिस्से में दर्द होता है?',
        'question_english': 'Do you experience pain in lower abdomen?',
        'hint_hindi': '"हां" या "नहीं"',
        'hint_english': '"Yes" or "No"',
        'type': 'yes_no'
    }
]

# PCOS Questions (extends PCOD with additional questions)
PCOS_QUESTIONS = PCOD_QUESTIONS[:8] + [  # First 8 from PCOD
    {
        'id': 'infertility',
        'question_hindi': 'क्या आपको गर्भधारण में कठिनाई हो रही है?',
        'question_english': 'Are you having difficulty conceiving?',
        'hint_hindi': '"हां" या "नहीं"',
        'hint_english': '"Yes" or "No"',
        'type': 'yes_no'
    },
    {
        'id': 'miscarriage',
        'question_hindi': 'आपको कितनी बार गर्भपात हुआ है?',
        'question_english': 'How many miscarriages have you had?',
        'hint_hindi': 'उदाहरण: "एक" या "0"',
        'hint_english': 'Example: "One" or "0"',
        'type': 'number',
        'validation': {'min': 0, 'max': 20}
    },
    PCOD_QUESTIONS[8]  # Pain question at the end
]


@voice_conversation_bp.route('/voice_selection')
def voice_selection():
    """Landing page to select PCOD or PCOS voice assessment"""
    if "user" not in session:
        flash("Please sign in to access voice assessment", "error")
        return redirect(url_for("signin"))
    return render_template('voice_selection.html')


@voice_conversation_bp.route('/voice_conversation_pcod')
def voice_conversation_pcod():
    """PCOD voice conversation chatbot"""
    if "user" not in session:
        flash("Please sign in to access chatbot", "error")
        return redirect(url_for("signin"))

    # Initialize session
    session['voice_assessment_type'] = 'PCOD'
    session['voice_current_question'] = 0
    session['voice_answers'] = {}
    session['voice_language'] = 'hindi'
    session.modified = True

    return render_template('voice_conversation_pcod.html', questions=PCOD_QUESTIONS)


@voice_conversation_bp.route('/voice_conversation_pcos')
def voice_conversation_pcos():
    """PCOS voice conversation chatbot"""
    if "user" not in session:
        flash("Please sign in to access chatbot", "error")
        return redirect(url_for("signin"))

    # Initialize session
    session['voice_assessment_type'] = 'PCOS'
    session['voice_current_question'] = 0
    session['voice_answers'] = {}
    session['voice_language'] = 'hindi'
    session.modified = True

    return render_template('voice_conversation_pcos.html', questions=PCOS_QUESTIONS)


@voice_conversation_bp.route('/voice/get_question', methods=['GET'])
def get_question():
    """Get current question"""
    try:
        assessment_type = session.get('voice_assessment_type', 'PCOD')
        current_index = session.get('voice_current_question', 0)
        language = session.get('voice_language', 'hindi')

        questions = PCOD_QUESTIONS if assessment_type == 'PCOD' else PCOS_QUESTIONS

        if current_index >= len(questions):
            return jsonify({'completed': True})

        question = questions[current_index]

        return jsonify({
            'success': True,
            'question_id': question['id'],
            'question': question[f'question_{language}'],
            'hint': question[f'hint_{language}'],
            'type': question['type'],
            'current': current_index + 1,
            'total': len(questions)
        })

    except Exception as e:
        logger.error(f"Error getting question: {e}")
        return jsonify({'error': str(e)}), 500


@voice_conversation_bp.route('/voice/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded voice audio"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        question_id = request.form.get('question_id')
        question_type = request.form.get('question_type')

        if not audio_file or audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Only allow WEBM, WAV, or OPUS files
        allowed_extensions = {'webm', 'wav', 'opus'}
        filename_ext = audio_file.filename.rsplit('.', 1)[-1].lower()
        if filename_ext not in allowed_extensions:
            return jsonify({'error': f'Invalid file type: {filename_ext}'}), 400

        # Save audio file
        filename = secure_filename(f"{uuid.uuid4()}.{filename_ext}")
        filepath = os.path.join(VOICE_UPLOAD_FOLDER, filename)
        audio_file.save(filepath)

        # Optional: Validate file size (<5MB)
        max_file_size = 5 * 1024 * 1024
        if os.path.getsize(filepath) > max_file_size:
            os.remove(filepath)
            return jsonify({'error': 'File too large (max 5MB)'}), 400

        # Transcribe
        language = session.get('voice_language', 'hindi')
        transcriber.set_language(language)
        transcribed_text = transcriber.transcribe_audio_file(filepath)

        # Clean up file
        try:
            os.remove(filepath)
        except Exception:
            pass

        if not transcribed_text:
            return jsonify({
                'success': False,
                'error': 'Could not transcribe audio. Please speak clearly and try again.'
            })

        # Process with NLP
        result = nlp_processor.process_answer(transcribed_text, question_type, question_id)

        if result['success']:
            # Generate confirmation
            confirmation = nlp_processor.generate_confirmation(
                question_id,
                result['value'],
                language
            )

            return jsonify({
                'success': True,
                'transcribed_text': transcribed_text,
                'extracted_value': result['value'],
                'confirmation': confirmation,
                'confidence': result['confidence']
            })
        else:
            return jsonify({
                'success': False,
                'transcribed_text': transcribed_text,
                'error': result.get('error', 'Could not understand your answer. Please try again.')
            })

    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        return jsonify({'error': 'Server error processing audio'}), 500



@voice_conversation_bp.route('/voice/save_answer', methods=['POST'])
def save_answer():
    """Save answer and move to next question"""
    try:
        data = request.get_json()
        question_id = data.get('question_id')
        value = data.get('value')

        if question_id is None or value is None:
            return jsonify({'error': 'Missing question_id or value'}), 400

        # Save answer
        if 'voice_answers' not in session:
            session['voice_answers'] = {}

        session['voice_answers'][question_id] = value
        session['voice_current_question'] = session.get('voice_current_question', 0) + 1
        session.modified = True

        # Check if completed
        assessment_type = session.get('voice_assessment_type', 'PCOD')
        questions = PCOD_QUESTIONS if assessment_type == 'PCOD' else PCOS_QUESTIONS

        if session['voice_current_question'] >= len(questions):
            # Submit to prediction
            return jsonify({'completed': True, 'redirect': '/voice/submit'})

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Error saving answer: {e}")
        return jsonify({'error': str(e)}), 500

@voice_conversation_bp.route('/voice/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get("language", "hindi")
    session['voice_language'] = lang
    return jsonify({"success": True, "language": lang})



@voice_conversation_bp.route('/voice/submit', methods=['GET', 'POST'])
def submit_voice_assessment():
    """Submit voice assessment and get prediction"""
    try:
        from app import load_users, save_assessment
        from app import predict_pcod, predict_pcos

        assessment_type = session.get('voice_assessment_type', 'PCOD')
        answers = session.get('voice_answers', {})

        if not answers:
            flash("No assessment data found", "error")
            return redirect(url_for('voice_conversation.voice_selection'))

        # Convert to form data format
        form_data = {}
        for key, value in answers.items():
            form_data[key] = value

        # Add stress and insulin as defaults (0) if not present
        if 'stress' not in form_data:
            form_data['stress'] = 0
        if 'insulin' not in form_data:
            form_data['insulin'] = 0

        # Create mock request with form data
        class MockRequest:
            def __init__(self, form_data):
                self.form = form_data

        mock_request = MockRequest(form_data)

        # Run prediction
        if assessment_type == 'PCOD':
            prediction, severity, probability = predict_pcod(mock_request)
        else:
            prediction, severity, probability = predict_pcos(mock_request)

        # Save assessment
        user_id = session.get("user")
        save_assessment(
            user_id,
            assessment_type,
            answers,
            prediction,
            severity,
            probability,
            via="voice"
        )

        # Show results page
        return render_template(
            "voice_results.html",
            assessment_type=assessment_type,
            answers=answers,
            prediction=prediction,
            severity=severity,
            probability=probability
        )

    except Exception as e:
        logger.error(f"Error submitting voice assessment: {e}", exc_info=True)
        flash("Server error while submitting voice assessment", "error")
        return redirect(url_for("voice_conversation.voice_selection"))
