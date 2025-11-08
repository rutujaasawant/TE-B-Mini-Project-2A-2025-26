# # import os
# # import logging
# # from flask import Flask, render_template, request, jsonify, session
# # from flask_cors import CORS
# # import pandas as pd
# # import numpy as np
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.preprocessing import StandardScaler
# # import pickle
# #
# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)
# # logger = logging.getLogger(__name__)
# #
# # app = Flask(__name__)
# # app.config['SECRET_KEY'] = 'your-secret-key'
# # app.permanent_session_lifetime = 1800  # 30 minutes
# # CORS(app)
# #
# #
# # # PCOS/PCOD Predictor Class
# # class PCOSPCODPredictor:
# #     def __init__(self):
# #         self.model = None
# #         self.scaler = None
# #         self.feature_names = []
# #         self.load_or_train_model()
# #
# #     def preprocess_data(self, df):
# #         for col in df.select_dtypes(include='object').columns:
# #             df[col] = df[col].str.lower()
# #         df.replace({'yes': 1, 'no': 0}, inplace=True)
# #         return df
# #
# #     def load_or_train_model(self):
# #         if os.path.exists('pcos_model.pkl') and os.path.exists('pcos_scaler.pkl'):
# #             try:
# #                 with open('pcos_model.pkl', 'rb') as f:
# #                     self.model = pickle.load(f)
# #                 with open('pcos_scaler.pkl', 'rb') as f:
# #                     self.scaler = pickle.load(f)
# #                 with open('feature_names.pkl', 'rb') as f:
# #                     self.feature_names = pickle.load(f)
# #                 logger.info("Models loaded successfully!")
# #             except Exception as e:
# #                 logger.error(f"Error loading models: {e}")
# #                 self.train_model()
# #         else:
# #             self.train_model()
# #
# #     def train_model(self):
# #         try:
# #             # Generate dummy data for testing
# #             df1 = pd.DataFrame({
# #                 'Age': np.random.randint(18, 45, 100),
# #                 'Length_of_cycle': np.random.randint(21, 35, 100),
# #                 'Length_of_menses': np.random.randint(2, 8, 100),
# #                 'Unusual_Bleeding': np.random.choice([0, 1], 100),
# #                 'BMI': np.random.uniform(18, 35, 100),
# #                 'Mean_of_length_of_cycle': np.random.randint(21, 35, 100),
# #                 'Menses_score': np.random.randint(1, 5, 100)
# #             })
# #             df3 = pd.DataFrame({
# #                 'Age': np.random.randint(18, 45, 100),
# #                 'BMI': np.random.uniform(18, 35, 100),
# #                 'Menstrual_Irregularity': np.random.choice([0, 1], 100),
# #                 'Testosterone_Level(ng/dL)': np.random.uniform(20, 80, 100),
# #                 'PCOS_Diagnosis': np.random.choice([0, 1], 100)
# #             })
# #
# #             df1 = self.preprocess_data(df1)
# #             df3 = self.preprocess_data(df3)
# #
# #             X_cols_df1 = ['Age', 'Length_of_cycle', 'Length_of_menses', 'Unusual_Bleeding', 'BMI',
# #                           'Mean_of_length_of_cycle', 'Menses_score']
# #             X_cols_df3 = ['Age', 'BMI', 'Menstrual_Irregularity', 'Testosterone_Level(ng/dL)']
# #
# #             df1_subset = df1[X_cols_df1].copy()
# #             df1_subset['target'] = 0
# #
# #             df3_subset = df3[X_cols_df3 + ['PCOS_Diagnosis']].copy()
# #             df3_subset['target'] = df3_subset['PCOS_Diagnosis']
# #             df3_subset = df3_subset.drop('PCOS_Diagnosis', axis=1)
# #
# #             X_train_data = []
# #             y_train_data = []
# #
# #             for _, row in df1_subset.iterrows():
# #                 features = [
# #                     row['Age'], row['BMI'], row['Length_of_cycle'], row['Length_of_menses'],
# #                     row['Unusual_Bleeding'], row['Mean_of_length_of_cycle'], row['Menses_score']
# #                 ]
# #                 X_train_data.append(features)
# #                 y_train_data.append(0)
# #
# #             for _, row in df3_subset.iterrows():
# #                 features = [row['Age'], row['BMI'], 30, 5, 0, 28, 3]
# #                 if 'Menstrual_Irregularity' in row:
# #                     features[2] = 35 if row['Menstrual_Irregularity'] == 1 else 28
# #                     features[4] = row['Menstrual_Irregularity']
# #                 X_train_data.append(features)
# #                 y_train_data.append(2 if row['target'] == 1 else 0)
# #
# #             X = np.array(X_train_data)
# #             y = np.array(y_train_data)
# #             self.feature_names = ['Age', 'BMI', 'Length_of_cycle', 'Length_of_menses', 'Unusual_Bleeding',
# #                                   'Mean_of_length_of_cycle', 'Menses_score']
# #
# #             self.scaler = StandardScaler()
# #             X_scaled = self.scaler.fit_transform(X)
# #
# #             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
# #             self.model.fit(X_scaled, y)
# #
# #             with open('pcos_model.pkl', 'wb') as f:
# #                 pickle.dump(self.model, f)
# #             with open('pcos_scaler.pkl', 'wb') as f:
# #                 pickle.dump(self.scaler, f)
# #             with open('feature_names.pkl', 'wb') as f:
# #                 pickle.dump(self.feature_names, f)
# #             logger.info("Model trained and saved successfully!")
# #         except Exception as e:
# #             logger.error(f"Error training model: {e}")
# #             self.feature_names = ['Age', 'BMI', 'Length_of_cycle', 'Length_of_menses', 'Unusual_Bleeding',
# #                                   'Mean_of_length_of_cycle', 'Menses_score']
# #             self.scaler = StandardScaler()
# #             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
# #             X_dummy = np.random.rand(100, 7)
# #             y_dummy = np.random.randint(0, 3, 100)
# #             X_scaled = self.scaler.fit_transform(X_dummy)
# #             self.model.fit(X_scaled, y_dummy)
# #             logger.info("Using dummy model for testing!")
# #
# #     def predict(self, features):
# #         logger.debug(f"Predicting with features: {features}")
# #         try:
# #             features_array = np.array(features).reshape(1, -1)
# #             features_scaled = self.scaler.transform(features_array)
# #             prediction = self.model.predict(features_scaled)[0]
# #             probability = self.model.predict_proba(features_scaled)[0]
# #             logger.debug(f"Prediction: {prediction}, Probability: {probability}")
# #             return prediction, probability
# #         except Exception as e:
# #             logger.error(f"Prediction error: {str(e)}", exc_info=True)
# #             return 0, [0.7, 0.2, 0.1]
# #
# #
# # predictor = PCOSPCODPredictor()
# #
# # # Updated QUESTIONS
# # QUESTIONS = [
# #     {'id': 'age', 'question': 'What is your age?', 'type': 'options',
# #      'options': ['18-25', '26-30', '31-35', '36-40', '40+']},
# #     {'id': 'bmi', 'question': 'What is your BMI (Body Mass Index)?', 'type': 'options',
# #      'options': ['Below 18.5 (Underweight)', '18.5-24.9 (Normal)', '25.0-29.9 (Overweight)',
# #                  '30.0-34.9 (Obese Class 1)', '35+ (Obese Class 2+)']},
# #     {'id': 'cycle_length', 'question': 'What is your typical menstrual cycle length?', 'type': 'options',
# #      'options': ['Less than 21 days', '21-28 days', '29-35 days', 'More than 35 days', 'Irregular/Unpredictable']},
# #     {'id': 'menses_length', 'question': 'How long does your menstrual period typically last?', 'type': 'options',
# #      'options': ['1-2 days', '3-5 days', '6-7 days', 'More than 7 days']},
# #     {'id': 'unusual_bleeding', 'question': 'Do you experience unusual bleeding between periods?', 'type': 'options',
# #      'options': ['Yes', 'No']},
# #     {'id': 'cycle_regularity', 'question': 'How regular are your menstrual cycles?', 'type': 'options',
# #      'options': ['Very regular (same length each month)', 'Mostly regular (1-2 days variation)', 'Somewhat irregular',
# #                  'Very irregular']},
# #     {'id': 'symptoms', 'question': 'Which symptoms do you experience? (Select the most prominent)', 'type': 'options',
# #      'options': ['No significant symptoms', 'Mild symptoms (occasional cramps)',
# #                  'Moderate symptoms (regular discomfort)', 'Severe symptoms (affects daily activities)']}
# # ]
# #
# #
# # def determine_condition_type(answers, prediction_level):
# #     logger.debug(f"Determining condition type: answers={answers}, prediction_level={prediction_level}")
# #
# #     if prediction_level == 0:
# #         return {"NORMAL": "Normal reproductive health"}
# #
# #     bmi_range = answers.get('bmi', '18.5-24.9 (Normal)')
# #     bmi_map = {
# #         'Below 18.5 (Underweight)': 17.0,
# #         '18.5-24.9 (Normal)': 22.0,
# #         '25.0-29.9 (Overweight)': 27.0,
# #         '30.0-34.9 (Obese Class 1)': 32.0,
# #         '35+ (Obese Class 2+)': 37.0
# #     }
# #     bmi = bmi_map.get(bmi_range, 22.0)
# #
# #     pcos_score = 0
# #     pcod_score = 0
# #
# #     age = answers.get('age', '18-25')
# #     if age in ['26-30', '31-35']:
# #         pcos_score += 1
# #     else:
# #         pcod_score += 1
# #
# #     if bmi >= 30:
# #         pcos_score += 2
# #     elif bmi >= 25:
# #         pcos_score += 1
# #     else:
# #         pcod_score += 1
# #
# #     cycle_length = answers.get('cycle_length', '21-28 days')
# #     cycle_regularity = answers.get('cycle_regularity', 'Very regular (same length each month)')
# #     if cycle_length == 'Irregular/Unpredictable' or cycle_regularity == 'Very irregular':
# #         pcos_score += 2
# #     elif cycle_length in ['More than 35 days', '29-35 days'] or cycle_regularity in ['Somewhat irregular']:
# #         if bmi >= 25:
# #             pcos_score += 1
# #         else:
# #             pcod_score += 1
# #     else:
# #         pcod_score += 1
# #
# #     symptoms = answers.get('symptoms', 'No significant symptoms')
# #     if symptoms == 'Severe symptoms (affects daily activities)':
# #         pcos_score += 2
# #     elif symptoms == 'Moderate symptoms (regular discomfort)':
# #         pcod_score += 1
# #
# #     if pcos_score == 0 and pcod_score == 0:
# #         return {"NORMAL": "Normal reproductive health"}
# #
# #     if pcos_score > pcod_score:
# #         if pcos_score <= 2:
# #             return {"PCOS": "Mild PCOS"}
# #         elif pcos_score <= 4:
# #             return {"PCOS": "Moderate PCOS"}
# #         else:
# #             return {"PCOS": "Severe PCOS"}
# #     elif pcod_score > pcos_score:
# #         if pcod_score <= 2:
# #             return {"PCOD": "Mild PCOD"}
# #         elif pcod_score <= 4:
# #             return {"PCOD": "Moderate PCOD"}
# #         else:
# #             return {"PCOD": "Severe PCOD"}
# #     else:
# #         return {"NORMAL": "Normal reproductive health"}
# #
# #
# # def process_answers(answers):
# #     logger.debug(f"Processing answers: {answers}")
# #     try:
# #         features = []
# #         age_map = {'18-25': 22, '26-30': 28, '31-35': 33, '36-40': 38, '40+': 42}
# #         features.append(age_map.get(answers.get('age', '18-25'), 22))
# #
# #         bmi_map = {
# #             'Below 18.5 (Underweight)': 17.0,
# #             '18.5-24.9 (Normal)': 22.0,
# #             '25.0-29.9 (Overweight)': 27.0,
# #             '30.0-34.9 (Obese Class 1)': 32.0,
# #             '35+ (Obese Class 2+)': 37.0
# #         }
# #         bmi = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
# #         features.append(bmi)
# #
# #         cycle_map = {
# #             'Less than 21 days': 18, '21-28 days': 28, '29-35 days': 32,
# #             'More than 35 days': 40, 'Irregular/Unpredictable': 35
# #         }
# #         features.append(cycle_map.get(answers.get('cycle_length', '21-28 days'), 28))
# #
# #         menses_map = {'1-2 days': 2, '3-5 days': 4, '6-7 days': 6, 'More than 7 days': 8}
# #         features.append(menses_map.get(answers.get('menses_length', '3-5 days'), 4))
# #
# #         bleeding_map = {'Yes': 1, 'No': 0}
# #         features.append(bleeding_map.get(answers.get('unusual_bleeding', 'No'), 0))
# #
# #         regularity_map = {
# #             'Very regular (same length each month)': 28,
# #             'Mostly regular (1-2 days variation)': 29,
# #             'Somewhat irregular': 32,
# #             'Very irregular': 38
# #         }
# #         features.append(
# #             regularity_map.get(answers.get('cycle_regularity', 'Very regular (same length each month)'), 28))
# #
# #         symptoms_map = {
# #             'No significant symptoms': 1,
# #             'Mild symptoms (occasional cramps)': 2,
# #             'Moderate symptoms (regular discomfort)': 3,
# #             'Severe symptoms (affects daily activities)': 5
# #         }
# #         features.append(symptoms_map.get(answers.get('symptoms', 'No significant symptoms'), 1))
# #
# #         logger.debug(f"Features: {features}, BMI: {bmi}")
# #         prediction, probability = predictor.predict(features)
# #         logger.debug(f"Prediction: {prediction}, Probability: {probability}")
# #         return prediction, probability
# #     except Exception as e:
# #         logger.error(f"Error processing answers: {str(e)}", exc_info=True)
# #         return 0, [0.7, 0.2, 0.1]
# #
# #
# # @app.route('/')
# # def index():
# #     return render_template('index.html')
# #
# #
# # @app.route('/chatbot')
# # def chatbot():
# #     session.clear()
# #     session.permanent = True
# #     session['current_question'] = 0
# #     session['answers'] = {}
# #     logger.debug("Chatbot session initialized")
# #     return render_template('chatbot_pcod.html')
# #
# #
# # @app.route('/start_chat', methods=['POST'])
# # def start_chat():
# #     session.permanent = True
# #     session['current_question'] = 0
# #     session['answers'] = {}
# #     session.modified = True
# #     response = {
# #         'question': QUESTIONS[0]['question'],
# #         'options': QUESTIONS[0]['options'],
# #         'question_id': QUESTIONS[0]['id']
# #     }
# #     return jsonify(response)
# #
# #
# # @app.route('/answer', methods=['POST'])
# # def answer():
# #     try:
# #         data = request.get_json()
# #         if not data:
# #             return jsonify({'error': 'No JSON data provided'}), 400
# #         question_id = data.get('question_id')
# #         answer_val = data.get('answer')
# #         if not question_id or not answer_val:
# #             return jsonify({'error': 'Missing question_id or answer'}), 400
# #
# #         if 'answers' not in session:
# #             session['answers'] = {}
# #         session['answers'][question_id] = answer_val
# #         session['current_question'] = session.get('current_question', 0) + 1
# #         session.modified = True
# #
# #         if session['current_question'] < len(QUESTIONS):
# #             next_question = QUESTIONS[session['current_question']]
# #             response = {
# #                 'question': next_question['question'],
# #                 'options': next_question['options'],
# #                 'question_id': next_question['id']
# #             }
# #             return jsonify(response)
# #         else:
# #             prediction, probability = process_answers(session['answers'])
# #             condition_type = determine_condition_type(session['answers'], prediction)
# #             session['prediction'] = int(prediction)
# #             session['probability'] = probability.tolist()
# #             session['condition_type'] = condition_type
# #             session.modified = True
# #             return jsonify({'completed': True, 'redirect': '/result'})
# #     except Exception as e:
# #         logger.error(f"Error in /answer: {str(e)}", exc_info=True)
# #         return jsonify({'error': 'Server error'}), 500
# #
# #
# # @app.route('/result')
# # def result():
# #     prediction = session.get('prediction', 0)
# #     probability = session.get('probability', [0.7, 0.2, 0.1])
# #     condition_type = session.get('condition_type', {"NORMAL": "Normal reproductive health"})
# #     answers = session.get('answers', {})
# #
# #     # Determine display info
# #     condition_name, severity = next(iter(condition_type.items()))
# #     color_map = {"NORMAL": "#28a745", "PCOD": "#ffc107", "PCOS": "#dc3545"}
# #     color = color_map.get(condition_name, "#28a745")
# #
# #     return render_template('results.html', prediction=prediction, condition_type=condition_name,
# #                            severity=severity, color=color, probability=probability, answers=answers)
# #
# #
# # @app.route('/doctors')
# # def doctor_selection():
# #     return render_template('doctor_selection.html')
# #
# #
# # @app.route('/voice-call')
# # @app.route('/voice-call/<doctor_id>')
# # def voice_call(doctor_id=None):
# #     return render_template('voice_call.html')
# #
# #
# # @app.route('/video-call')
# # @app.route('/video-call/<doctor_id>')
# # def video_call(doctor_id=None):
# #     return render_template('video_call.html')
# #
# #
# # @app.route('/health-info')
# # def health_info():
# #     return render_template('health_info.html')
# #
# #
# # @app.route('/test-calls')
# # def test_calls():
# #     return render_template('test_calls.html')
# #
# #
# # if __name__ == '__main__':
# #     logger.info("Starting PCOS/PCOD Health Application...")
# #     app.run(debug=False, host='0.0.0.0', port=5000)
#
#
# import os
# import logging
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import pickle
#
# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key'
# app.permanent_session_lifetime = 1800  # 30 minutes
# CORS(app)
#
# # Define QUESTIONS for the chatbot, aligned with dataset features
# QUESTIONS = [
#     {
#         'id': 'age',
#         'question': 'What is your age range?',
#         'options': ['18-25', '26-30', '31-35', '36-40', '40+']
#     },
#     {
#         'id': 'bmi',
#         'question': 'What is your BMI range?',
#         'options': ['Below 18.5 (Underweight)', '18.5-24.9 (Normal)', '25.0-29.9 (Overweight)',
#                     '30.0-34.9 (Obese Class 1)', '35+ (Obese Class 2+)']
#     },
#     {
#         'id': 'cycle_length',
#         'question': 'What is the average length of your menstrual cycle?',
#         'options': ['Less than 21 days', '21-28 days', '29-35 days', 'More than 35 days', 'Irregular/Unpredictable']
#     },
#     {
#         'id': 'menses_length',
#         'question': 'How long do your periods typically last?',
#         'options': ['1-2 days', '3-5 days', '6-7 days', 'More than 7 days']
#     },
#     {
#         'id': 'unusual_bleeding',
#         'question': 'Do you experience abnormal bleeding between periods?',
#         'options': ['Yes', 'No']
#     },
#     {
#         'id': 'cycle_regularity',
#         'question': 'How regular are your menstrual cycles?',
#         'options': ['Very regular (same length each month)', 'Mostly regular (1-2 days variation)',
#                     'Somewhat irregular', 'Very irregular']
#     },
#     {
#         'id': 'symptoms',
#         'question': 'Do you experience symptoms like pelvic pain or cramps?',
#         'options': ['None', 'Occasional (mild)', 'Frequent (moderate)', 'Severe (affects daily life)']
#     },
#     {
#         'id': 'hormonal_issues',
#         'question': 'Do you have signs of hormonal issues (e.g., excessive hair growth, acne, or medical conditions like diabetes or thyroid issues)?',
#         'options': ['None', 'Mild (e.g., occasional acne)', 'Moderate (e.g., noticeable hair growth)',
#                     'Severe (e.g., diagnosed diabetes or thyroid issues)']
#     }
# ]
#
#
# # PCOS/PCOD Predictor Class
# class PCOSPCODPredictor:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.feature_names = []
#         self.load_or_train_model()
#
#     def preprocess_data(self, df):
#         for col in df.select_dtypes(include='object').columns:
#             df[col] = df[col].str.lower()
#         df.replace({'yes': 1, 'no': 0}, inplace=True)
#         return df
#
#     def load_or_train_model(self):
#         if os.path.exists('pcos_model.pkl') and os.path.exists('pcos_scaler.pkl'):
#             try:
#                 with open('pcos_model.pkl', 'rb') as f:
#                     self.model = pickle.load(f)
#                 with open('pcos_scaler.pkl', 'rb') as f:
#                     self.scaler = pickle.load(f)
#                 with open('feature_names.pkl', 'rb') as f:
#                     self.feature_names = pickle.load(f)
#                 logger.info("Models loaded successfully!")
#             except Exception as e:
#                 logger.error(f"Error loading models: {e}")
#                 self.train_model()
#         else:
#             self.train_model()
#
#     def train_model(self):
#         try:
#             # Load datasets
#             df_pcos = pd.read_csv('pcos_dataset.csv')
#             df_result = pd.read_csv('result_data.csv')
#             df_excel = pd.read_excel('PCOS_data_without_infertility.xlsx', sheet_name='Full_new')
#
#             # Preprocess datasets
#             df_pcos = self.preprocess_data(df_pcos)
#             df_result = self.preprocess_data(df_result)
#             df_excel = self.preprocess_data(df_excel)
#
#             # Define features
#             self.feature_names = ['Age', 'BMI', 'Length_of_cycle', 'Length_of_menses', 'Unusual_Bleeding',
#                                   'Mean_of_length_of_cycle', 'Menses_score']
#
#             # Prepare training data
#             X_train_data = []
#             y_train_data = []
#
#             # Process pcos_dataset.csv
#             for _, row in df_pcos.iterrows():
#                 features = [
#                     row['Age'],
#                     row['BMI'],
#                     35 if row['Menstrual_Irregularity'] == 1 else 28,  # Length_of_cycle
#                     5,  # Length_of_menses (default)
#                     row['Menstrual_Irregularity'],  # Unusual_Bleeding proxy
#                     28,  # Mean_of_length_of_cycle (default)
#                     3  # Menses_score (default)
#                 ]
#                 # Label: 0 (Normal), 1 (PCOD), 2 (PCOS)
#                 if row['PCOS_Diagnosis'] == 1:
#                     label = 2 if row['Testosterone_Level(ng/dL)'] > 50 else 1
#                 else:
#                     label = 0
#
#                 X_train_data.append(features)
#                 y_train_data.append(label)
#
#             # Process PCOS_data_without_infertility.xlsx
#             for _, row in df_excel.iterrows():
#                 features = [
#                     row['Age (yrs)'],
#                     row['BMI'],
#                     35 if row['Cycle(R/I)'] == 4 else 28,  # Irregular cycles map to longer length
#                     5,  # Length_of_menses (default)
#                     1 if row['Cycle(R/I)'] == 4 else 0,  # Unusual_Bleeding proxy
#                     28,  # Mean_of_length_of_cycle (default)
#                     3  # Menses_score (default)
#                 ]
#                 # Label: 0 (Normal), 1 (PCOD), 2 (PCOS)
#                 if row['PCOS (Y/N)'] == 1:
#                     label = 2 if row['FSH/LH'] > 2 else 1  # Use FSH/LH ratio for PCOS distinction
#                 else:
#                     label = 0
#                 X_train_data.append(features)
#                 y_train_data.append(label)
#
#             # Process result_data.csv
#             symptoms_map = {
#                 'never': 1,
#                 'sometime': 2,
#                 'sometimes': 2,
#                 'always': 4
#             }
#             for _, row in df_result.iterrows():
#                 age = 30  # Default age
#                 bmi = 25  # Default BMI
#                 cycle_length = 35 if row.get('Signs of PCOD', '').lower() in ['abnormal uterine bleeding',
#                                                                               'all of the above'] else 28
#                 menses_score = symptoms_map.get(
#                     row.get('7) Do you experience any pain and cramps in your lower abdomen, lower back and legs?',
#                             'never').lower(), 1)
#                 hormonal_score = 3 if row.get('Medical Condition', '').lower() in ['diabetes', 'thyroid',
#                                                                                    'hypertension'] else \
#                     2 if row.get('Signs of PCOD', '').lower() in ['hirsutism', 'all of the above'] else 0
#
#                 features = [
#                     age,
#                     bmi,
#                     cycle_length,
#                     5,  # Length_of_menses (default)
#                     1 if row.get('Signs of PCOD', '').lower() in ['abnormal uterine bleeding',
#                                                                   'all of the above'] else 0,
#                     cycle_length,  # Mean_of_length_of_cycle (proxy)
#                     menses_score + hormonal_score  # Combine symptoms and hormonal issues
#                 ]
#                 # Label: 0 (Normal), 1 (PCOD), 2 (PCOS)
#                 if row.get('Signs of PCOD', '').lower() in ['hirsutism', 'abnormal uterine bleeding',
#                                                             'all of the above']:
#                     label = 2 if row.get('Medical Condition', '').lower() in ['diabetes', 'thyroid',
#                                                                               'hypertension'] else 1
#                 else:
#                     label = 0
#                 X_train_data.append(features)
#                 y_train_data.append(label)
#
#             X = np.array(X_train_data)
#             y = np.array(y_train_data)
#
#             self.scaler = StandardScaler()
#             X_scaled = self.scaler.fit_transform(X)
#
#             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#             self.model.fit(X_scaled, y)
#
#             with open('pcos_model.pkl', 'wb') as f:
#                 pickle.dump(self.model, f)
#             with open('pcos_scaler.pkl', 'wb') as f:
#                 pickle.dump(self.scaler, f)
#             with open('feature_names.pkl', 'wb') as f:
#                 pickle.dump(self.feature_names, f)
#             logger.info("Model trained and saved successfully!")
#         except Exception as e:
#             logger.error(f"Error training model: {e}")
#             # Fallback to dummy data
#             df1 = pd.DataFrame({
#                 'Age': np.random.randint(18, 45, 100),
#                 'Length_of_cycle': np.random.randint(21, 35, 100),
#                 'Length_of_menses': np.random.randint(2, 8, 100),
#                 'Unusual_Bleeding': np.random.choice([0, 1], 100),
#                 'BMI': np.random.uniform(18, 35, 100),
#                 'Mean_of_length_of_cycle': np.random.randint(21, 35, 100),
#                 'Menses_score': np.random.randint(1, 5, 100)
#             })
#             df1['target'] = np.random.choice([0, 1, 2], 100)  # Random labels for dummy data
#             X = df1[self.feature_names].values
#             y = df1['target'].values
#             self.scaler = StandardScaler()
#             X_scaled = self.scaler.fit_transform(X)
#             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#             self.model.fit(X_scaled, y)
#             logger.info("Using dummy model for testing!")
#
#     def predict(self, features):
#         logger.debug(f"Predicting with features: {features}")
#         try:
#             features_array = np.array(features).reshape(1, -1)
#             features_scaled = self.scaler.transform(features_array)
#             prediction = self.model.predict(features_scaled)[0]
#             probability = self.model.predict_proba(features_scaled)[0]
#             logger.debug(f"Prediction: {prediction}, Probability: {probability}")
#             return prediction, probability
#         except Exception as e:
#             logger.error(f"Error in predict: {str(e)}")
#             return 0, [0.7, 0.2, 0.1]
#
#
# predictor = PCOSPCODPredictor()
#
#
# def rule_based_adjustment(answers, predicted_label):
#     """
#     Override prediction based on simple clinical rules for PCOD/PCOS.
#     Especially useful if ML predicts Normal but symptoms strongly suggest PCOD/PCOS.
#     """
#
#     # Map answers to severity scores
#     symptoms_map = {
#         'None': 0,
#         'Occasional (mild)': 1,
#         'Frequent (moderate)': 2,
#         'Severe (affects daily life)': 3
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#
#     symptom_score = symptoms_map.get(answers.get('symptoms', 'None'), 0)
#     hormonal_score = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#
#     # Also check cycle irregularity
#     cycle_irregular = answers.get('cycle_length', '') in ['Irregular/Unpredictable', 'More than 35 days']
#     unusual_bleeding = answers.get('unusual_bleeding', '') == 'Yes'
#     cycle_irregularity = answers.get('cycle_regularity', '') in ['Somewhat irregular', 'Very irregular']
#
#     # Simple heuristic rules:
#     # 1. If ML predicts Normal but cycles irregular + moderate symptoms + low hormonal issues => PCOD
#     if predicted_label == 0:
#         if cycle_irregular and unusual_bleeding and cycle_irregularity and symptom_score >= 1 and hormonal_score <= 1:
#             return 1  # Override to PCOD only
#
#         # 2. If cycles irregular + moderate to severe hormones => PCOS
#         if cycle_irregular and hormonal_score >= 2:
#             return 2
#
#     return predicted_label
#
#
#
# def process_answers(answers):
#     logger.debug(f"Processing answers: {answers}")
#     try:
#         features = []
#         age_map = {'18-25': 22, '26-30': 28, '31-35': 33, '36-40': 38, '40+': 42}
#         features.append(age_map.get(answers.get('age', '18-25'), 22))
#
#         bmi_map = {
#             'Below 18.5 (Underweight)': 17.0,
#             '18.5-24.9 (Normal)': 22.0,
#             '25.0-29.9 (Overweight)': 27.0,
#             '30.0-34.9 (Obese Class 1)': 32.0,
#             '35+ (Obese Class 2+)': 37.0
#         }
#         bmi = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
#         features.append(bmi)
#
#         cycle_map = {
#             'Less than 21 days': 18, '21-28 days': 28, '29-35 days': 32,
#             'More than 35 days': 40, 'Irregular/Unpredictable': 35
#         }
#         features.append(cycle_map.get(answers.get('cycle_length', '21-28 days'), 28))
#
#         menses_map = {'1-2 days': 2, '3-5 days': 4, '6-7 days': 6, 'More than 7 days': 8}
#         features.append(menses_map.get(answers.get('menses_length', '3-5 days'), 4))
#
#         bleeding_map = {'Yes': 1, 'No': 0}
#         features.append(bleeding_map.get(answers.get('unusual_bleeding', 'No'), 0))
#
#         regularity_map = {
#             'Very regular (same length each month)': 28,
#             'Mostly regular (1-2 days variation)': 29,
#             'Somewhat irregular': 32,
#             'Very irregular': 38
#         }
#         features.append(
#             regularity_map.get(answers.get('cycle_regularity', 'Very regular (same length each month)'), 28))
#
#         symptoms_map = {
#             'None': 1,
#             'Occasional (mild)': 2,
#             'Frequent (moderate)': 3,
#             'Severe (affects daily life)': 5
#         }
#         hormonal_map = {
#             'None': 0,
#             'Mild (e.g., occasional acne)': 1,
#             'Moderate (e.g., noticeable hair growth)': 2,
#             'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#         }
#         menses_score = symptoms_map.get(answers.get('symptoms', 'None'), 1) + \
#                        hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#         features.append(menses_score)
#
#         logger.debug(f"Features: {features}, BMI: {bmi}, Menses_score: {menses_score}")
#         prediction, probability = predictor.predict(features)
#         logger.debug(f"Prediction: {prediction}, Probability: {probability}")
#         return prediction, probability
#     except Exception as e:
#         logger.error(f"Error processing answers: {str(e)}", exc_info=True)
#         return 0, [0.7, 0.2, 0.1]
#
#
# def get_intensity(symptom_val, hormonal_val):
#     score = symptom_val + hormonal_val
#     if score >= 6:
#         return "High"
#     elif score >= 3:
#         return "Medium"
#     return "Low"
#
# def determine_condition_type(answers, prediction):
#     symptoms_map = {
#         'None': 1,
#         'Occasional (mild)': 2,
#         'Frequent (moderate)': 3,
#         'Severe (affects daily life)': 5
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#     symptom_val = symptoms_map.get(answers.get('symptoms', 'None'), 1)
#     hormonal_val = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#     intensity = get_intensity(symptom_val, hormonal_val)
#
#     if prediction == 2:
#         return {"condition": "PCOS (includes PCOD features)", "intensity": intensity,
#                 "description": "Likely PCOS: cystic ovaries and systemic hormonal issues.",
#                 "medical_advice": "Consult a specialist for hormonal management."}
#     elif prediction == 1:
#         return {"condition": "PCOD only", "intensity": intensity,
#                 "description": "Likely PCOD: cystic ovaries, normal hormones.",
#                 "medical_advice": "Manage with diet, exercise."}
#     else:
#         return {"condition": "Normal", "intensity": "Low",
#                 "description": "No detected disorder. Healthy reproductive pattern.",
#                 "medical_advice": "Maintain routine care."}
#
#
#
#
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/chatbot')
# def chatbot():
#     session.clear()
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     logger.debug("Chatbot session initialized")
#     return render_template('chatbot_pcod.html')
#
#
# @app.route('/start_chat', methods=['POST'])
# def start_chat():
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     response = {
#         'question': QUESTIONS[0]['question'],
#         'options': QUESTIONS[0]['options'],
#         'question_id': QUESTIONS[0]['id']
#     }
#     return jsonify(response)
#
#
# @app.route('/answer', methods=['POST'])
# def answer():
#     try:
#         data = request.get_json()
#         if not data:
#             logger.error("No JSON data provided in /answer")
#             return jsonify({'error': 'No JSON data provided'}), 400
#         question_id = data.get('question_id')
#         answer_val = data.get('answer')
#         if not question_id or not answer_val:
#             logger.error(f"Missing question_id or answer: {data}")
#             return jsonify({'error': 'Missing question_id or answer'}), 400
#
#         if 'answers' not in session:
#             session['answers'] = {}
#         session['answers'][question_id] = answer_val
#         session['current_question'] = session.get('current_question', 0) + 1
#         session.modified = True
#         logger.debug(f"Answer recorded: {question_id} = {answer_val}, Current question: {session['current_question']}")
#
#         if session['current_question'] < len(QUESTIONS):
#             next_question = QUESTIONS[session['current_question']]
#             response = {
#                 'question': next_question['question'],
#                 'options': next_question['options'],
#                 'question_id': next_question['id']
#             }
#             return jsonify(response)
#         else:
#             prediction, probability = process_answers(session['answers'])
#             adjusted_prediction = rule_based_adjustment(session['answers'], prediction)
#             condition_type = determine_condition_type(session['answers'], adjusted_prediction)
#             condition_type = determine_condition_type(session['answers'], prediction)
#             session['prediction'] = int(prediction)
#             session['probability'] = probability.tolist()
#             session['condition_type'] = condition_type
#             session.modified = True
#             logger.debug(f"Session data set: prediction={prediction}, condition_type={condition_type}")
#             return jsonify({'completed': True, 'redirect': '/result'})
#     except Exception as e:
#         logger.error(f"Error in /answer: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Server error'}), 500
#
#
# @app.route('/result')
# def result():
#     try:
#         prediction = session.get('prediction', 0)
#         probability = session.get('probability', [0.7, 0.2, 0.1])
#         condition_type = session.get('condition_type', {
#             'condition': 'Normal',
#             'intensity': 'Low',
#             'description': 'Normal reproductive health',
#             'medical_advice': 'Continue maintaining healthy lifestyle habits.'
#         })
#         answers = session.get('answers', {})
#
#         color_map = {"Normal": "#28a745", "PCOD": "#ffc107", "PCOS": "#dc3545"}
#         color = color_map.get(condition_type['condition'], "#28a745")
#
#         logger.debug(
#             f"Rendering result: condition={condition_type['condition']}, intensity={condition_type['intensity']}")
#         return render_template('results.html',
#                                prediction=prediction,
#                                condition_type=condition_type.get('condition', 'Normal'),
#                                severity=condition_type.get('intensity', 'Low'),
#                                description=condition_type.get('description', 'Normal reproductive health'),
#                                medical_advice=condition_type.get('medical_advice',
#                                                                  'Continue maintaining healthy lifestyle habits.'),
#                                probability=probability,
#                                answers=answers,
#                                color=color)
#
#     except Exception as e:
#         logger.error(f"Error in /result: {str(e)}", exc_info=True)
#         return redirect(url_for('chatbot'))
#
#
#
# @app.route('/doctors')
# def doctor_selection():
#     return render_template('doctor_selection.html')
#
#
# @app.route('/voice-call')
# @app.route('/voice-call/<doctor_id>')
# def voice_call(doctor_id=None):
#     return render_template('voice_call.html')
#
#
# @app.route('/video-call')
# @app.route('/video-call/<doctor_id>')
# def video_call(doctor_id=None):
#     return render_template('video_call.html')
#
#
# @app.route('/health-info')
# def health_info():
#     return render_template('health_info.html')
#
#
# @app.route('/test-calls')
# def test_calls():
#     return render_template('test_calls.html')
#
#
# if __name__ == '__main__':
#     logger.info("Starting PCOS/PCOD Health Application...")
#     app.run(debug=True, host='0.0.0.0', port=5000)
#
#
#
#



# import os
# import uuid
# import json
# import logging
# from datetime import datetime
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
#
# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key'
# app.permanent_session_lifetime = 1800  # 30 minutes
# CORS(app)
#
# ASSESSMENTS_FILE = 'assessments.csv'
#
# # Define QUESTIONS for the chatbot, aligned with dataset features
# QUESTIONS = [
#     {
#         'id': 'age',
#         'question': 'What is your age range?',
#         'options': ['18-25', '26-30', '31-35', '36-40', '40+']
#     },
#     {
#         'id': 'bmi',
#         'question': 'What is your BMI range?',
#         'options': ['Below 18.5 (Underweight)', '18.5-24.9 (Normal)', '25.0-29.9 (Overweight)', '30.0-34.9 (Obese Class 1)', '35+ (Obese Class 2+)']
#     },
#     {
#         'id': 'cycle_length',
#         'question': 'What is the average length of your menstrual cycle?',
#         'options': ['Less than 21 days', '21-28 days', '29-35 days', 'More than 35 days', 'Irregular/Unpredictable']
#     },
#     {
#         'id': 'menses_length',
#         'question': 'How long do your periods typically last?',
#         'options': ['1-2 days', '3-5 days', '6-7 days', 'More than 7 days']
#     },
#     {
#         'id': 'unusual_bleeding',
#         'question': 'Do you experience abnormal bleeding between periods?',
#         'options': ['Yes', 'No']
#     },
#     {
#         'id': 'cycle_regularity',
#         'question': 'How regular are your menstrual cycles?',
#         'options': ['Very regular (same length each month)', 'Mostly regular (1-2 days variation)', 'Somewhat irregular', 'Very irregular']
#     },
#     {
#         'id': 'symptoms',
#         'question': 'Do you experience symptoms like pelvic pain or cramps?',
#         'options': ['None', 'Occasional (mild)', 'Frequent (moderate)', 'Severe (affects daily life)']
#     },
#     {
#         'id': 'hormonal_issues',
#         'question': 'Do you have signs of hormonal issues (e.g., excessive hair growth, acne, or medical conditions like diabetes or thyroid issues)?',
#         'options': ['None', 'Mild (e.g., occasional acne)', 'Moderate (e.g., noticeable hair growth)', 'Severe (e.g., diagnosed diabetes or thyroid issues)']
#     }
# ]
#
# # Utility: calculate intensity label from symptom/hormonal scores
# def get_intensity_label(symptom_score, hormonal_score):
#     total = symptom_score + hormonal_score
#     if total >= 6:
#         return "High"
#     elif total >= 3:
#         return "Medium"
#     return "Low"
#
# # Convert questionnaire answers to numeric scores used by the heuristics
# def score_answers(answers):
#     # symptom/hormonal scoring
#     symptoms_map = {
#         'None': 0,
#         'Occasional (mild)': 1,
#         'Frequent (moderate)': 2,
#         'Severe (affects daily life)': 3
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#     symptom_score = symptoms_map.get(answers.get('symptoms', 'None'), 0)
#     hormonal_score = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#
#     cycle_len = answers.get('cycle_length', '21-28 days')
#     cycle_irregular = cycle_len in ['Irregular/Unpredictable', 'More than 35 days']
#
#     unusual_bleeding = answers.get('unusual_bleeding', 'No') == 'Yes'
#     reg = answers.get('cycle_regularity', 'Very regular (same length each month)')
#     reg_score = 0
#     if reg == 'Very regular (same length each month)':
#         reg_score = 0
#     elif reg == 'Mostly regular (1-2 days variation)':
#         reg_score = 1
#     elif reg == 'Somewhat irregular':
#         reg_score = 2
#     else:
#         reg_score = 3
#
#     bmi_map = {
#         'Below 18.5 (Underweight)': 17.0,
#         '18.5-24.9 (Normal)': 22.0,
#         '25.0-29.9 (Overweight)': 27.0,
#         '30.0-34.9 (Obese Class 1)': 32.0,
#         '35+ (Obese Class 2+)': 37.0
#     }
#     bmi_val = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
#
#     return {
#         'symptom_score': symptom_score,
#         'hormonal_score': hormonal_score,
#         'cycle_irregular': cycle_irregular,
#         'unusual_bleeding': unusual_bleeding,
#         'reg_score': reg_score,
#         'bmi_val': bmi_val
#     }
#
# # -------- Rule-based predictor ----------
# def rule_based_predict(answers):
#     s = score_answers(answers)
#     symptom = s['symptom_score']
#     hormonal = s['hormonal_score']
#     cycle_irregular = s['cycle_irregular']
#     unusual_bleeding = s['unusual_bleeding']
#     reg_score = s['reg_score']
#     bmi_val = s['bmi_val']
#
#     # default Normal
#     label = 0
#     probs = np.array([0.85, 0.05, 0.05, 0.05])
#
#     # --- Priority: PCOS with PCOD first ---
#     if hormonal >= 2 and cycle_irregular and (unusual_bleeding or symptom >= 2):
#         label = 3
#         probs = np.array([0.02, 0.05, 0.45, 0.48])
#     # --- Then PCOS ---
#     elif hormonal >= 2 and (cycle_irregular or bmi_val >= 30):
#         label = 2
#         probs = np.array([0.05, 0.05, 0.7, 0.2])
#     # --- Then PCOD ---
#     elif (cycle_irregular or reg_score >= 2) and hormonal <= 1 and symptom >= 1:
#         label = 1
#         probs = np.array([0.1, 0.6, 0.15, 0.15])
#     # --- Else stays Normal ---
#
#     # Adjust probabilities slightly
#     total_score = symptom + hormonal + reg_score
#     factor = min(total_score / 10.0, 0.4)
#     if label == 0:
#         probs = probs * (1 - factor) + np.array([1, 0, 0, 0]) * factor
#     elif label == 1:
#         probs = probs * (1 - factor) + np.array([0, 1, 0, 0]) * factor
#     elif label == 2:
#         probs = probs * (1 - factor) + np.array([0, 0, 1, 0]) * factor
#     elif label == 3:
#         probs = probs * (1 - factor) + np.array([0, 0, 0, 1]) * factor
#
#     probs = probs / probs.sum()
#     intensity = get_intensity_label(symptom, hormonal)
#     return int(label), probs.tolist(), intensity
#
#
#
# # Save or update single CSV file with the record for this session
# def save_or_update_record(record_id, answers, label, probs, intensity):
#     """
#     Ensures exactly one CSV file (ASSESSMENTS_FILE) exists.
#     If a record with record_id exists, update it. Otherwise append.
#     """
#     df_row = {
#         'record_id': record_id,
#         'timestamp': datetime.utcnow().isoformat(),
#         'answers_json': json.dumps(answers, ensure_ascii=False),
#         'prediction_label': int(label),
#         'prediction_text': label_to_text(label),
#         'intensity': intensity,
#         'prob_0_normal': float(probs[0]) if len(probs) > 0 else 0.0,
#         'prob_1_pcod': float(probs[1]) if len(probs) > 1 else 0.0,
#         'prob_2_pcos': float(probs[2]) if len(probs) > 2 else 0.0,
#         'prob_3_pcos_pcod': float(probs[3]) if len(probs) > 3 else 0.0
#     }
#     cols = list(df_row.keys())
#     if os.path.exists(ASSESSMENTS_FILE):
#         try:
#             df = pd.read_csv(ASSESSMENTS_FILE)
#             if 'record_id' not in df.columns:
#                 df = pd.DataFrame(columns=cols)
#             if record_id in df['record_id'].astype(str).values:
#                 df.loc[df['record_id'].astype(str) == str(record_id), :] = pd.Series(df_row)
#             else:
#                 df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
#             df.to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessment saved/updated to existing file {ASSESSMENTS_FILE}")
#         except Exception as e:
#             logger.error(f"Error updating assessments file: {e}", exc_info=True)
#             try:
#                 pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#                 logger.info(f"Assessments file overwritten due to error; new file {ASSESSMENTS_FILE} created.")
#             except Exception as e2:
#                 logger.error(f"Failed to write assessments file: {e2}", exc_info=True)
#     else:
#         try:
#             pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessments file {ASSESSMENTS_FILE} created and record saved.")
#         except Exception as e:
#             logger.error(f"Failed to create assessments file: {e}", exc_info=True)
#
# def label_to_text(label):
#     return {
#         0: "Normal",
#         1: "PCOS only",
#         2: "PCOD only",
#         3: "PCOS with PCOD"
#     }.get(int(label), "Normal")
#
# # --------- Flask routes ----------
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/chatbot')
# def chatbot():
#     session.clear()
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     logger.debug("Chatbot session initialized (rule-based version)")
#     return render_template('chatbot_pcod.html')
#
# @app.route('/start_chat', methods=['POST'])
# def start_chat():
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     response = {
#         'question': QUESTIONS[0]['question'],
#         'options': QUESTIONS[0]['options'],
#         'question_id': QUESTIONS[0]['id']
#     }
#     return jsonify(response)
#
# @app.route('/answer', methods=['POST'])
# def answer():
#     try:
#         data = request.get_json()
#         if not data:
#             logger.error("No JSON data provided in /answer")
#             return jsonify({'error': 'No JSON data provided'}), 400
#         question_id = data.get('question_id')
#         answer_val = data.get('answer')
#         if not question_id or (answer_val is None):
#             logger.error(f"Missing question_id or answer: {data}")
#             return jsonify({'error': 'Missing question_id or answer'}), 400
#         if 'answers' not in session:
#             session['answers'] = {}
#         session['answers'][question_id] = answer_val
#         session['current_question'] = session.get('current_question', 0) + 1
#         session.modified = True
#         logger.debug(f"Answer recorded: {question_id} = {answer_val}, Current question: {session['current_question']}")
#
#         if session['current_question'] < len(QUESTIONS):
#             next_question = QUESTIONS[session['current_question']]
#             response = {
#                 'question': next_question['question'],
#                 'options': next_question['options'],
#                 'question_id': next_question['id']
#             }
#             return jsonify(response)
#         else:
#             answers = session.get('answers', {})
#             label, probs, intensity = rule_based_predict(answers)
#             if 'record_id' not in session:
#                 session['record_id'] = str(uuid.uuid4())
#             record_id = session['record_id']
#             try:
#                 save_or_update_record(record_id, answers, label, probs, intensity)
#             except Exception as e:
#                 logger.error(f"Error saving/updating record: {e}", exc_info=True)
#
#             session['prediction'] = int(label)
#             session['probability'] = probs
#             session['intensity'] = intensity
#             session['condition_type'] = {
#                 'condition': label_to_text(label),
#                 'intensity': intensity,
#                 'description': _get_description_for_label(label),
#                 'medical_advice': _get_advice_for_label(label, intensity)
#             }
#             session.modified = True
#             logger.debug(f"Session result set: label={label}, intensity={intensity}")
#             return jsonify({'completed': True, 'redirect': '/result'})
#     except Exception as e:
#         logger.error(f"Error in /answer: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Server error'}), 500
#
# def _get_description_for_label(label):
#     if label == 0:
#         return "No detected disorder. Normal reproductive function likely."
#     elif label == 1:
#         return "Likely PCOD: predominantly irregular cycles without strong systemic hormonal disturbance."
#     elif label == 2:
#         return "Likely PCOS: systemic hormonal imbalance with features such as hirsutism, metabolic risk, and irregular cycles."
#     elif label == 3:
#         return "Likely combined PCOS with PCOD features: both significant hormonal imbalance and ovarian cystic features."
#     return "Normal reproductive health."
#
# def _get_advice_for_label(label, intensity):
#     if label == 0:
#         return "Maintain a healthy lifestyle. If you have concerns, consult a gynecologist for routine checkups."
#     if label == 1:
#         base = "Consider lifestyle measures (diet, exercise). Consult gynecologist for pelvic ultrasound and cycle management."
#     elif label == 2:
#         base = "See an endocrinologist/gynecologist for hormonal evaluation. Consider metabolic screening (glucose, lipids)."
#     elif label == 3:
#         base = "Seek specialist care (endocrinology + gynecology). Comprehensive hormonal and imaging evaluation recommended."
#     if intensity == "High":
#         return base + " Symptoms severe — seek prompt specialist evaluation."
#     elif intensity == "Medium":
#         return base + " Moderate symptoms — schedule a specialist visit."
#     else:
#         return base + " Mild symptoms — monitor and follow-up if symptoms progress."
#
# @app.route('/result')
# def result():
#     try:
#         prediction = session.get('prediction', 0)
#         probability = session.get('probability', [0.85, 0.05, 0.05, 0.05])
#         condition_type = session.get('condition_type', {
#             'condition': 'Normal',
#             'intensity': 'Low',
#             'description': 'Normal reproductive health',
#             'medical_advice': 'Continue maintaining healthy lifestyle habits.'
#         })
#
#         # color map
#         color_map = {
#             "Normal": "#28a745",
#             "PCOD only": "#ffc107",
#             "PCOS only": "#dc3545",
#             "PCOS with PCOD": "#8b008b"
#         }
#         color = color_map.get(condition_type['condition'], "#28a745")
#
#         return render_template(
#             "results.html",
#             prediction=prediction,
#             probability=probability,
#             condition_type=condition_type['condition'],
#             risk_level=f"{condition_type['intensity']} Risk",
#             severity=condition_type['intensity'],
#             description=condition_type['description'],
#             medical_advice=condition_type['medical_advice'],
#             color=color
#         )
#     except Exception as e:
#         logger.error(f"Error in /result: {str(e)}", exc_info=True)
#         return redirect(url_for('chatbot'))
#
#
# @app.route('/doctors')
# def doctor_selection():
#     return render_template('doctor_selection.html')
#
# @app.route('/voice-call')
# @app.route('/voice-call/<doctor_id>')
# def voice_call(doctor_id=None):
#     return render_template('voice_call.html')
#
# @app.route('/video-call')
# @app.route('/video-call/<doctor_id>')
# def video_call(doctor_id=None):
#     return render_template('video_call.html')
#
# @app.route('/health-info')
# def health_info():
#     return render_template('health_info.html')
#
# @app.route('/test-calls')
# def test_calls():
#     return render_template('test_calls.html')
#
# if __name__ == '__main__':
#     logger.info("Starting rule-based PCOS/PCOD Health Application...")
#     app.run(debug=True, host='0.0.0.0', port=5000)


# import os
# import uuid
# import json
# import logging
# from datetime import datetime
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
#
# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key'
# app.permanent_session_lifetime = 1800  # 30 minutes
# CORS(app)
#
# ASSESSMENTS_FILE = 'assessments.csv'
#
# # Load the pre-trained PCOS logistic regression and scaler models
# # Ensure these files exist in your app directory, or retrain as needed.
# with open('logreg_pcos.pkl', 'rb') as f:
#     logreg_pcos = pickle.load(f)
#
# with open('scaler_pcos.pkl', 'rb') as f:
#     scaler_pcos = pickle.load(f)
#
# # Your existing questions unchanged
# QUESTIONS = [
#     {
#         'id': 'age',
#         'question': 'What is your age range?',
#         'options': ['18-25', '26-30', '31-35', '36-40', '40+']
#     },
#     {
#         'id': 'bmi',
#         'question': 'What is your BMI range?',
#         'options': ['Below 18.5 (Underweight)', '18.5-24.9 (Normal)', '25.0-29.9 (Overweight)',
#                     '30.0-34.9 (Obese Class 1)', '35+ (Obese Class 2+)']
#     },
#     {
#         'id': 'cycle_length',
#         'question': 'What is the average length of your menstrual cycle?',
#         'options': ['Less than 21 days', '21-28 days', '29-35 days', 'More than 35 days', 'Irregular/Unpredictable']
#     },
#     {
#         'id': 'menses_length',
#         'question': 'How long do your periods typically last?',
#         'options': ['1-2 days', '3-5 days', '6-7 days', 'More than 7 days']
#     },
#     {
#         'id': 'unusual_bleeding',
#         'question': 'Do you experience abnormal bleeding between periods?',
#         'options': ['Yes', 'No']
#     },
#     {
#         'id': 'cycle_regularity',
#         'question': 'How regular are your menstrual cycles?',
#         'options': ['Very regular (same length each month)', 'Mostly regular (1-2 days variation)',
#                     'Somewhat irregular', 'Very irregular']
#     },
#     {
#         'id': 'symptoms',
#         'question': 'Do you experience symptoms like pelvic pain or cramps?',
#         'options': ['None', 'Occasional (mild)', 'Frequent (moderate)', 'Severe (affects daily life)']
#     },
#     {
#         'id': 'hormonal_issues',
#         'question': 'Do you have signs of hormonal issues (e.g., excessive hair growth, acne, or medical conditions like diabetes or thyroid issues)?',
#         'options': ['None', 'Mild (e.g., occasional acne)', 'Moderate (e.g., noticeable hair growth)',
#                     'Severe (e.g., diagnosed diabetes or thyroid issues)']
#     }
# ]
#
#
# # Preprocess functions for encoding the features for PCOS prediction using ML model
# def preprocess_for_ml(answers):
#     # Age mapping to numeric (approximate midpoint of ranges)
#     age_map = {
#         '18-25': 21,
#         '26-30': 28,
#         '31-35': 33,
#         '36-40': 38,
#         '40+': 45
#     }
#     age_val = age_map.get(answers.get('age', '18-25'), 21)
#
#     # BMI mapping to midpoint of range
#     bmi_map = {
#         'Below 18.5 (Underweight)': 17.0,
#         '18.5-24.9 (Normal)': 22.0,
#         '25.0-29.9 (Overweight)': 27.0,
#         '30.0-34.9 (Obese Class 1)': 32.0,
#         '35+ (Obese Class 2+)': 37.0
#     }
#     bmi_val = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
#
#     # Menstrual irregularity: convert cycle_length to 0 or 1
#     # Assume irregular if 'Irregular/Unpredictable' or 'More than 35 days'
#     cycle_length = answers.get('cycle_length', '21-28 days')
#     menstrual_irregularity = 1 if cycle_length in ['Irregular/Unpredictable', 'More than 35 days'] else 0
#
#     # Testosterone level and Antral Follicle Count are not in questionnaire—use proxy scores from symptoms/hormonal issues
#     # Use symptom and hormonal intensity scoring similar to existing method to approximate levels
#     symptoms_map = {
#         'None': 0,
#         'Occasional (mild)': 1,
#         'Frequent (moderate)': 2,
#         'Severe (affects daily life)': 3
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#     symptom_score = symptoms_map.get(answers.get('symptoms', 'None'), 0)
#     hormonal_score = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#
#     # Approximate Testosterone_Level(ng/dL) and Antral_Follicle_Count from scores
#     testosterone_level = 20 + 10 * hormonal_score  # arbitrary scaling
#     antral_follicle_count = 10 + 10 * symptom_score  # arbitrary scaling
#
#     # Create feature vector as dataframe for scaler/model
#     features = pd.DataFrame([[
#         age_val,
#         bmi_val,
#         menstrual_irregularity,
#         testosterone_level,
#         antral_follicle_count
#     ]], columns=['Age', 'BMI', 'Menstrual_Irregularity',
#                  'Testosterone_Level(ng/dL)', 'Antral_Follicle_Count'])
#
#     # Scale features
#     scaled_features = scaler_pcos.transform(features)
#
#     return scaled_features
#
#
# # Existing score_answers and rule_based_predict unchanged
#
# def get_intensity_label(symptom_score, hormonal_score):
#     total = symptom_score + hormonal_score
#     if total >= 6:
#         return "High"
#     elif total >= 3:
#         return "Medium"
#     return "Low"
#
#
# def score_answers(answers):
#     symptoms_map = {
#         'None': 0,
#         'Occasional (mild)': 1,
#         'Frequent (moderate)': 2,
#         'Severe (affects daily life)': 3
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#     symptom_score = symptoms_map.get(answers.get('symptoms', 'None'), 0)
#     hormonal_score = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#
#     cycle_len = answers.get('cycle_length', '21-28 days')
#     cycle_irregular = cycle_len in ['Irregular/Unpredictable', 'More than 35 days']
#
#     unusual_bleeding = answers.get('unusual_bleeding', 'No') == 'Yes'
#     reg = answers.get('cycle_regularity', 'Very regular (same length each month)')
#     reg_score = 0
#     if reg == 'Very regular (same length each month)':
#         reg_score = 0
#     elif reg == 'Mostly regular (1-2 days variation)':
#         reg_score = 1
#     elif reg == 'Somewhat irregular':
#         reg_score = 2
#     else:
#         reg_score = 3
#
#     bmi_map = {
#         'Below 18.5 (Underweight)': 17.0,
#         '18.5-24.9 (Normal)': 22.0,
#         '25.0-29.9 (Overweight)': 27.0,
#         '30.0-34.9 (Obese Class 1)': 32.0,
#         '35+ (Obese Class 2+)': 37.0
#     }
#     bmi_val = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
#
#     return {
#         'symptom_score': symptom_score,
#         'hormonal_score': hormonal_score,
#         'cycle_irregular': cycle_irregular,
#         'unusual_bleeding': unusual_bleeding,
#         'reg_score': reg_score,
#         'bmi_val': bmi_val
#     }
#
#
# def label_to_text(label):
#     return {
#         0: "Normal",
#         1: "PCOD only",
#         2: "PCOS only",
#         3: "PCOS with PCOD"
#     }.get(int(label), "Normal")
#
#
# # Combined prediction function using ML for PCOS and rule-based for PCOD
# def combined_predict(answers):
#     # ML-based PCOS prediction
#     X_ml = preprocess_for_ml(answers)
#     pcos_prob = logreg_pcos.predict_proba(X_ml)[0]
#     pcos_pred = int(np.argmax(pcos_prob))
#
#     # Rule-based PCOD prediction (your existing rule_based_predict function)
#     rb_label, rb_probs, intensity = rule_based_predict(answers)
#
#     # Combine predictions heuristically:
#     # If ML PCOS predicts positive with high confidence (>0.6), use PCOS label (2 or 3)
#     # Otherwise, use rule-based PCOD label (0 or 1) but updated with intensity
#
#     # Possible combined labels:
#     # 0: Normal; 1: PCOD only; 2: PCOS only; 3: PCOS with PCOD
#
#     if pcos_pred == 1:  # ML says PCOS positive
#         # Decide if combined or pure PCOS based on rule-based PCOD label
#         if rb_label in [1, 3]:  # rule-based says some PCOD
#             combined_label = 3  # PCOS with PCOD
#         else:
#             combined_label = 2  # PCOS only
#         combined_probs = [0, 0, pcos_prob[1], 0]  # simplified probs for ML PCOS
#         combined_probs[combined_label] = max(pcos_prob[1],
#                                              rb_probs[combined_label] if combined_label < len(rb_probs) else 0)
#     else:
#         # ML says Non-PCOS, trust rule-based for PCOD vs Normal
#         combined_label = rb_label if rb_label in [0, 1] else 0
#         combined_probs = rb_probs
#     intensity_combined = intensity
#
#     return combined_label, combined_probs, intensity_combined
#
#
# # Save or update single CSV file with the record for this session
# def save_or_update_record(record_id, answers, label, probs, intensity):
#     df_row = {
#         'record_id': record_id,
#         'timestamp': datetime.utcnow().isoformat(),
#         'answers_json': json.dumps(answers, ensure_ascii=False),
#         'prediction_label': int(label),
#         'prediction_text': label_to_text(label),
#         'intensity': intensity,
#         'prob_0_normal': float(probs[0]) if len(probs) > 0 else 0.0,
#         'prob_1_pcod': float(probs[1]) if len(probs) > 1 else 0.0,
#         'prob_2_pcos': float(probs[2]) if len(probs) > 2 else 0.0,
#         'prob_3_pcos_pcod': float(probs[3]) if len(probs) > 3 else 0.0
#     }
#     cols = list(df_row.keys())
#     if os.path.exists(ASSESSMENTS_FILE):
#         try:
#             df = pd.read_csv(ASSESSMENTS_FILE)
#             if 'record_id' not in df.columns:
#                 df = pd.DataFrame(columns=cols)
#             if record_id in df['record_id'].astype(str).values:
#                 df.loc[df['record_id'].astype(str) == str(record_id), :] = pd.Series(df_row)
#             else:
#                 df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
#             df.to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessment saved/updated to existing file {ASSESSMENTS_FILE}")
#         except Exception as e:
#             logger.error(f"Error updating assessments file: {e}", exc_info=True)
#             try:
#                 pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#                 logger.info(f"Assessments file overwritten due to error; new file {ASSESSMENTS_FILE} created.")
#             except Exception as e2:
#                 logger.error(f"Failed to write assessments file: {e2}", exc_info=True)
#     else:
#         try:
#             pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessments file {ASSESSMENTS_FILE} created and record saved.")
#         except Exception as e:
#             logger.error(f"Failed to create assessments file: {e}", exc_info=True)
#
#
# # Existing Flask routes remain unchanged except /answer uses combined_predict
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/chatbot')
# def chatbot():
#     session.clear()
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     logger.debug("Chatbot session initialized (combined ML + rule-based version)")
#     return render_template('chatbot_pcod.html')
#
#
# @app.route('/start_chat', methods=['POST'])
# def start_chat():
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     response = {
#         'question': QUESTIONS[0]['question'],
#         'options': QUESTIONS[0]['options'],
#         'question_id': QUESTIONS[0]['id']
#     }
#     return jsonify(response)
#
#
# @app.route('/answer', methods=['POST'])
# def answer():
#     try:
#         data = request.get_json()
#         if not data:
#             logger.error("No JSON data provided in /answer")
#             return jsonify({'error': 'No JSON data provided'}), 400
#         question_id = data.get('question_id')
#         answer_val = data.get('answer')
#         if not question_id or (answer_val is None):
#             logger.error(f"Missing question_id or answer: {data}")
#             return jsonify({'error': 'Missing question_id or answer'}), 400
#         if 'answers' not in session:
#             session['answers'] = {}
#         session['answers'][question_id] = answer_val
#         session['current_question'] = session.get('current_question', 0) + 1
#         session.modified = True
#         logger.debug(f"Answer recorded: {question_id} = {answer_val}, Current question: {session['current_question']}")
#
#         if session['current_question'] < len(QUESTIONS):
#             next_question = QUESTIONS[session['current_question']]
#             response = {
#                 'question': next_question['question'],
#                 'options': next_question['options'],
#                 'question_id': next_question['id']
#             }
#             return jsonify(response)
#         else:
#             answers = session.get('answers', {})
#             # Use combined ML + rule-based prediction
#             label, probs, intensity = combined_predict(answers)
#             if 'record_id' not in session:
#                 session['record_id'] = str(uuid.uuid4())
#             record_id = session['record_id']
#             try:
#                 save_or_update_record(record_id, answers, label, probs, intensity)
#             except Exception as e:
#                 logger.error(f"Error saving/updating record: {e}", exc_info=True)
#
#             session['prediction'] = int(label)
#             session['probability'] = probs
#             session['intensity'] = intensity
#             session['condition_type'] = {
#                 'condition': label_to_text(label),
#                 'intensity': intensity,
#                 'description': _get_description_for_label(label),
#                 'medical_advice': _get_advice_for_label(label, intensity)
#             }
#             session.modified = True
#             logger.debug(f"Session result set: label={label}, intensity={intensity}")
#             return jsonify({'completed': True, 'redirect': '/result'})
#     except Exception as e:
#         logger.error(f"Error in /answer: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Server error'}), 500
#
#
# def _get_description_for_label(label):
#     if label == 0:
#         return "No detected disorder. Normal reproductive function likely."
#     elif label == 1:
#         return "Likely PCOD: predominantly irregular cycles without strong systemic hormonal disturbance."
#     elif label == 2:
#         return "Likely PCOS: systemic hormonal imbalance with features such as hirsutism, metabolic risk, and irregular cycles."
#     elif label == 3:
#         return "Likely combined PCOS with PCOD features: both significant hormonal imbalance and ovarian cystic features."
#     return "Normal reproductive health."
#
#
# def _get_advice_for_label(label, intensity):
#     if label == 0:
#         return "Maintain a healthy lifestyle. If you have concerns, consult a gynecologist for routine checkups."
#     if label == 1:
#         base = "Consider lifestyle measures (diet, exercise). Consult gynecologist for pelvic ultrasound and cycle management."
#     elif label == 2:
#         base = "See an endocrinologist/gynecologist for hormonal evaluation. Consider metabolic screening (glucose, lipids)."
#     elif label == 3:
#         base = "Seek specialist care (endocrinology + gynecology). Comprehensive hormonal and imaging evaluation recommended."
#     if intensity == "High":
#         return base + " Symptoms severe — seek prompt specialist evaluation."
#     elif intensity == "Medium":
#         return base + " Moderate symptoms — schedule a specialist visit."
#     else:
#         return base + " Mild symptoms — monitor and follow-up if symptoms progress."
#
#
# @app.route('/result')
# def result():
#     try:
#         prediction = session.get('prediction', 0)
#         probability = session.get('probability', [0.85, 0.05, 0.05, 0.05])
#         condition_type = session.get('condition_type', {
#             'condition': 'Normal',
#             'intensity': 'Low',
#             'description': 'Normal reproductive health',
#             'medical_advice': 'Continue maintaining healthy lifestyle habits.'
#         })
#
#         # color map
#         color_map = {
#             "Normal": "#28a745",
#             "PCOD only": "#ffc107",
#             "PCOS only": "#dc3545",
#             "PCOS with PCOD": "#8b008b"
#         }
#         color = color_map.get(condition_type['condition'], "#28a745")
#
#         return render_template(
#             "results.html",
#             prediction=prediction,
#             probability=probability,
#             condition_type=condition_type['condition'],
#             risk_level=f"{condition_type['intensity']} Risk",
#             severity=condition_type['intensity'],
#             description=condition_type['description'],
#             medical_advice=condition_type['medical_advice'],
#             color=color
#         )
#     except Exception as e:
#         logger.error(f"Error in /result: {str(e)}", exc_info=True)
#         return redirect(url_for('chatbot'))
#
#
# # Other routes like /doctors, /voice-call, /video-call, /health-info unchanged
#
# if __name__ == '__main__':
#     logger.info("Starting combined ML + rule-based PCOS/PCOD Health Application...")
#     app.run(debug=True, host='0.0.0.0', port=5000)


# import os
# import uuid
# import json
# import logging
# from datetime import datetime
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
#
# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key'
# app.permanent_session_lifetime = 1800  # 30 minutes
# CORS(app)
#
# ASSESSMENTS_FILE = 'assessments.csv'
#
# # Load the pre-trained PCOS logistic regression and scaler models (optional)
# try:
#     with open('pcos_model.pkl', 'rb') as f:
#         logreg_pcos = pickle.load(f)
#     with open('pcos_scaler.pkl', 'rb') as f:
#         scaler_pcos = pickle.load(f)
#     logger.info("ML models loaded successfully")
#     USE_ML = True
# except FileNotFoundError:
#     logger.warning("ML model files not found. Using rule-based prediction only.")
#     logreg_pcos = None
#     scaler_pcos = None
#     USE_ML = False
#
# # Questions
# QUESTIONS = [
#     {
#         'id': 'age',
#         'question': 'What is your age range?',
#         'options': ['18-25', '26-30', '31-35', '36-40', '40+']
#     },
#     {
#         'id': 'bmi',
#         'question': 'What is your BMI range?',
#         'options': ['Below 18.5 (Underweight)', '18.5-24.9 (Normal)', '25.0-29.9 (Overweight)',
#                     '30.0-34.9 (Obese Class 1)', '35+ (Obese Class 2+)']
#     },
#     {
#         'id': 'cycle_length',
#         'question': 'What is the average length of your menstrual cycle?',
#         'options': ['Less than 21 days', '21-28 days', '29-35 days', 'More than 35 days', 'Irregular/Unpredictable']
#     },
#     {
#         'id': 'menses_length',
#         'question': 'How long do your periods typically last?',
#         'options': ['1-2 days', '3-5 days', '6-7 days', 'More than 7 days']
#     },
#     {
#         'id': 'unusual_bleeding',
#         'question': 'Do you experience abnormal bleeding between periods?',
#         'options': ['Yes', 'No']
#     },
#     {
#         'id': 'cycle_regularity',
#         'question': 'How regular are your menstrual cycles?',
#         'options': ['Very regular (same length each month)', 'Mostly regular (1-2 days variation)',
#                     'Somewhat irregular', 'Very irregular']
#     },
#     {
#         'id': 'symptoms',
#         'question': 'Do you experience symptoms like pelvic pain or cramps?',
#         'options': ['None', 'Occasional (mild)', 'Frequent (moderate)', 'Severe (affects daily life)']
#     },
#     {
#         'id': 'hormonal_issues',
#         'question': 'Do you have signs of hormonal issues (e.g., excessive hair growth, acne, or medical conditions like diabetes or thyroid issues)?',
#         'options': ['None', 'Mild (e.g., occasional acne)', 'Moderate (e.g., noticeable hair growth)',
#                     'Severe (e.g., diagnosed diabetes or thyroid issues)']
#     }
# ]
#
#
# # Preprocess functions for encoding the features for PCOS prediction using ML model
# def preprocess_for_ml(answers):
#     age_map = {
#         '18-25': 21,
#         '26-30': 28,
#         '31-35': 33,
#         '36-40': 38,
#         '40+': 45
#     }
#     age_val = age_map.get(answers.get('age', '18-25'), 21)
#
#     bmi_map = {
#         'Below 18.5 (Underweight)': 17.0,
#         '18.5-24.9 (Normal)': 22.0,
#         '25.0-29.9 (Overweight)': 27.0,
#         '30.0-34.9 (Obese Class 1)': 32.0,
#         '35+ (Obese Class 2+)': 37.0
#     }
#     bmi_val = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
#
#     cycle_length = answers.get('cycle_length', '21-28 days')
#     menstrual_irregularity = 1 if cycle_length in ['Irregular/Unpredictable', 'More than 35 days'] else 0
#
#     symptoms_map = {
#         'None': 0,
#         'Occasional (mild)': 1,
#         'Frequent (moderate)': 2,
#         'Severe (affects daily life)': 3
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#     symptom_score = symptoms_map.get(answers.get('symptoms', 'None'), 0)
#     hormonal_score = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#
#     # Approximate hormonal and metabolic features
#     testosterone_level = 20 + 10 * hormonal_score  # ng/dL (range ~20-50)
#     antral_follicle_count = 10 + 10 * symptom_score  # count (range ~10-40)
#
#     # Additional features likely in your model:
#     # LH/FSH ratio - approximate from hormonal issues (range ~0.5-3.0)
#     lh_fsh_ratio = 0.8 + 0.4 * hormonal_score
#
#     # Hirsutism score - approximate from hormonal issues (range 0-10)
#     hirsutism_score = hormonal_score * 2.5
#
#     # Model expects 7 features - try common PCOS dataset feature order:
#     # Age, BMI, Menstrual_Irregularity, Testosterone, Antral_Follicle_Count, LH_FSH_Ratio, Hirsutism
#     features = pd.DataFrame([[
#         age_val,
#         bmi_val,
#         menstrual_irregularity,
#         testosterone_level,
#         antral_follicle_count,
#         lh_fsh_ratio,
#         hirsutism_score
#     ]], columns=['Age', 'BMI', 'Menstrual_Irregularity', 'Testosterone_Level(ng/dL)',
#                  'Antral_Follicle_Count', 'LH_FSH_Ratio', 'Hirsutism_Score'])
#
#     scaled_features = scaler_pcos.transform(features)
#     return scaled_features
#
#
# def get_intensity_label(symptom_score, hormonal_score):
#     total = symptom_score + hormonal_score
#     if total >= 6:
#         return "High"
#     elif total >= 3:
#         return "Medium"
#     return "Low"
#
#
# def score_answers(answers):
#     symptoms_map = {
#         'None': 0,
#         'Occasional (mild)': 1,
#         'Frequent (moderate)': 2,
#         'Severe (affects daily life)': 3
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#     symptom_score = symptoms_map.get(answers.get('symptoms', 'None'), 0)
#     hormonal_score = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#
#     cycle_len = answers.get('cycle_length', '21-28 days')
#     cycle_irregular = cycle_len in ['Irregular/Unpredictable', 'More than 35 days']
#
#     unusual_bleeding = answers.get('unusual_bleeding', 'No') == 'Yes'
#     reg = answers.get('cycle_regularity', 'Very regular (same length each month)')
#     reg_score = 0
#     if reg == 'Very regular (same length each month)':
#         reg_score = 0
#     elif reg == 'Mostly regular (1-2 days variation)':
#         reg_score = 1
#     elif reg == 'Somewhat irregular':
#         reg_score = 2
#     else:
#         reg_score = 3
#
#     bmi_map = {
#         'Below 18.5 (Underweight)': 17.0,
#         '18.5-24.9 (Normal)': 22.0,
#         '25.0-29.9 (Overweight)': 27.0,
#         '30.0-34.9 (Obese Class 1)': 32.0,
#         '35+ (Obese Class 2+)': 37.0
#     }
#     bmi_val = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
#
#     return {
#         'symptom_score': symptom_score,
#         'hormonal_score': hormonal_score,
#         'cycle_irregular': cycle_irregular,
#         'unusual_bleeding': unusual_bleeding,
#         'reg_score': reg_score,
#         'bmi_val': bmi_val
#     }
#
#
# # Rule-based predictor (added from commented code)
# def rule_based_predict(answers):
#     s = score_answers(answers)
#     symptom = s['symptom_score']
#     hormonal = s['hormonal_score']
#     cycle_irregular = s['cycle_irregular']
#     unusual_bleeding = s['unusual_bleeding']
#     reg_score = s['reg_score']
#     bmi_val = s['bmi_val']
#
#     label = 0
#     probs = np.array([0.85, 0.05, 0.05, 0.05])
#
#     # Priority: PCOS with PCOD first
#     if hormonal >= 2 and cycle_irregular and (unusual_bleeding or symptom >= 2):
#         label = 3
#         probs = np.array([0.02, 0.05, 0.45, 0.48])
#     # Then PCOS
#     elif hormonal >= 2 and (cycle_irregular or bmi_val >= 30):
#         label = 2
#         probs = np.array([0.05, 0.05, 0.7, 0.2])
#     # Then PCOD
#     elif (cycle_irregular or reg_score >= 2) and hormonal <= 1 and symptom >= 1:
#         label = 1
#         probs = np.array([0.1, 0.6, 0.15, 0.15])
#
#     # Adjust probabilities
#     total_score = symptom + hormonal + reg_score
#     factor = min(total_score / 10.0, 0.4)
#     if label == 0:
#         probs = probs * (1 - factor) + np.array([1, 0, 0, 0]) * factor
#     elif label == 1:
#         probs = probs * (1 - factor) + np.array([0, 1, 0, 0]) * factor
#     elif label == 2:
#         probs = probs * (1 - factor) + np.array([0, 0, 1, 0]) * factor
#     elif label == 3:
#         probs = probs * (1 - factor) + np.array([0, 0, 0, 1]) * factor
#
#     probs = probs / probs.sum()
#     intensity = get_intensity_label(symptom, hormonal)
#     return int(label), probs.tolist(), intensity
#
#
# def label_to_text(label):
#     return {
#         0: "Normal",
#         1: "PCOD only",
#         2: "PCOS only",
#         3: "PCOS with PCOD"
#     }.get(int(label), "Normal")
#
#
# # Combined prediction function using ML for PCOS and rule-based for PCOD
# def combined_predict(answers):
#     # If ML models not available, use rule-based only
#     if not USE_ML or logreg_pcos is None or scaler_pcos is None:
#         logger.debug("Using rule-based prediction only")
#         return rule_based_predict(answers)
#
#     # ML-based PCOS prediction
#     X_ml = preprocess_for_ml(answers)
#     pcos_prob = logreg_pcos.predict_proba(X_ml)[0]
#     pcos_pred = int(np.argmax(pcos_prob))
#
#     rb_label, rb_probs, intensity = rule_based_predict(answers)
#
#     if pcos_pred == 1:
#         if rb_label in [1, 3]:
#             combined_label = 3
#         else:
#             combined_label = 2
#         combined_probs = [0, 0, pcos_prob[1], 0]
#         combined_probs[combined_label] = max(pcos_prob[1],
#                                              rb_probs[combined_label] if combined_label < len(rb_probs) else 0)
#     else:
#         combined_label = rb_label if rb_label in [0, 1] else 0
#         combined_probs = rb_probs
#
#     intensity_combined = intensity
#     return combined_label, combined_probs, intensity_combined
#
#
# def save_or_update_record(record_id, answers, label, probs, intensity):
#     df_row = {
#         'record_id': record_id,
#         'timestamp': datetime.utcnow().isoformat(),
#         'answers_json': json.dumps(answers, ensure_ascii=False),
#         'prediction_label': int(label),
#         'prediction_text': label_to_text(label),
#         'intensity': intensity,
#         'prob_0_normal': float(probs[0]) if len(probs) > 0 else 0.0,
#         'prob_1_pcod': float(probs[1]) if len(probs) > 1 else 0.0,
#         'prob_2_pcos': float(probs[2]) if len(probs) > 2 else 0.0,
#         'prob_3_pcos_pcod': float(probs[3]) if len(probs) > 3 else 0.0
#     }
#     cols = list(df_row.keys())
#     if os.path.exists(ASSESSMENTS_FILE):
#         try:
#             df = pd.read_csv(ASSESSMENTS_FILE)
#             if 'record_id' not in df.columns:
#                 df = pd.DataFrame(columns=cols)
#             if record_id in df['record_id'].astype(str).values:
#                 df.loc[df['record_id'].astype(str) == str(record_id), :] = pd.Series(df_row)
#             else:
#                 df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
#             df.to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessment saved/updated to existing file {ASSESSMENTS_FILE}")
#         except Exception as e:
#             logger.error(f"Error updating assessments file: {e}", exc_info=True)
#             try:
#                 pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#                 logger.info(f"Assessments file overwritten due to error; new file {ASSESSMENTS_FILE} created.")
#             except Exception as e2:
#                 logger.error(f"Failed to write assessments file: {e2}", exc_info=True)
#     else:
#         try:
#             pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessments file {ASSESSMENTS_FILE} created and record saved.")
#         except Exception as e:
#             logger.error(f"Failed to create assessments file: {e}", exc_info=True)
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/chatbot')
# def chatbot():
#     session.clear()
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     logger.debug("Chatbot session initialized (combined ML + rule-based version)")
#     return render_template('chatbot_pcod.html')
#
#
# @app.route('/start_chat', methods=['POST'])
# def start_chat():
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     response = {
#         'question': QUESTIONS[0]['question'],
#         'options': QUESTIONS[0]['options'],
#         'question_id': QUESTIONS[0]['id']
#     }
#     return jsonify(response)
#
#
# @app.route('/answer', methods=['POST'])
# def answer():
#     try:
#         data = request.get_json()
#         if not data:
#             logger.error("No JSON data provided in /answer")
#             return jsonify({'error': 'No JSON data provided'}), 400
#         question_id = data.get('question_id')
#         answer_val = data.get('answer')
#         if not question_id or (answer_val is None):
#             logger.error(f"Missing question_id or answer: {data}")
#             return jsonify({'error': 'Missing question_id or answer'}), 400
#         if 'answers' not in session:
#             session['answers'] = {}
#         session['answers'][question_id] = answer_val
#         session['current_question'] = session.get('current_question', 0) + 1
#         session.modified = True
#         logger.debug(f"Answer recorded: {question_id} = {answer_val}, Current question: {session['current_question']}")
#
#         if session['current_question'] < len(QUESTIONS):
#             next_question = QUESTIONS[session['current_question']]
#             response = {
#                 'question': next_question['question'],
#                 'options': next_question['options'],
#                 'question_id': next_question['id']
#             }
#             return jsonify(response)
#         else:
#             answers = session.get('answers', {})
#             label, probs, intensity = combined_predict(answers)
#             if 'record_id' not in session:
#                 session['record_id'] = str(uuid.uuid4())
#             record_id = session['record_id']
#             try:
#                 save_or_update_record(record_id, answers, label, probs, intensity)
#             except Exception as e:
#                 logger.error(f"Error saving/updating record: {e}", exc_info=True)
#
#             session['prediction'] = int(label)
#             session['probability'] = probs
#             session['intensity'] = intensity
#             session['condition_type'] = {
#                 'condition': label_to_text(label),
#                 'intensity': intensity,
#                 'description': _get_description_for_label(label),
#                 'medical_advice': _get_advice_for_label(label, intensity)
#             }
#             session.modified = True
#             logger.debug(f"Session result set: label={label}, intensity={intensity}")
#             return jsonify({'completed': True, 'redirect': '/result'})
#     except Exception as e:
#         logger.error(f"Error in /answer: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Server error'}), 500
#
#
# def _get_description_for_label(label):
#     if label == 0:
#         return "No detected disorder. Normal reproductive function likely."
#     elif label == 1:
#         return "Likely PCOD: predominantly irregular cycles without strong systemic hormonal disturbance."
#     elif label == 2:
#         return "Likely PCOS: systemic hormonal imbalance with features such as hirsutism, metabolic risk, and irregular cycles."
#     elif label == 3:
#         return "Likely combined PCOS with PCOD features: both significant hormonal imbalance and ovarian cystic features."
#     return "Normal reproductive health."
#
#
# def _get_advice_for_label(label, intensity):
#     if label == 0:
#         return "Maintain a healthy lifestyle. If you have concerns, consult a gynecologist for routine checkups."
#     if label == 1:
#         base = "Consider lifestyle measures (diet, exercise). Consult gynecologist for pelvic ultrasound and cycle management."
#     elif label == 2:
#         base = "See an endocrinologist/gynecologist for hormonal evaluation. Consider metabolic screening (glucose, lipids)."
#     elif label == 3:
#         base = "Seek specialist care (endocrinology + gynecology). Comprehensive hormonal and imaging evaluation recommended."
#     if intensity == "High":
#         return base + " Symptoms severe — seek prompt specialist evaluation."
#     elif intensity == "Medium":
#         return base + " Moderate symptoms — schedule a specialist visit."
#     else:
#         return base + " Mild symptoms — monitor and follow-up if symptoms progress."
#
#
# @app.route('/result')
# def result():
#     try:
#         prediction = session.get('prediction', 0)
#         probability = session.get('probability', [0.85, 0.05, 0.05, 0.05])
#         condition_type = session.get('condition_type', {
#             'condition': 'Normal',
#             'intensity': 'Low',
#             'description': 'Normal reproductive health',
#             'medical_advice': 'Continue maintaining healthy lifestyle habits.'
#         })
#
#         color_map = {
#             "Normal": "#28a745",
#             "PCOD only": "#ffc107",
#             "PCOS only": "#dc3545",
#             "PCOS with PCOD": "#8b008b"
#         }
#         color = color_map.get(condition_type['condition'], "#28a745")
#
#         return render_template(
#             "results.html",
#             prediction=prediction,
#             probability=probability,
#             condition_type=condition_type['condition'],
#             risk_level=f"{condition_type['intensity']} Risk",
#             severity=condition_type['intensity'],
#             description=condition_type['description'],
#             medical_advice=condition_type['medical_advice'],
#             color=color
#         )
#     except Exception as e:
#         logger.error(f"Error in /result: {str(e)}", exc_info=True)
#         return redirect(url_for('chatbot'))
#
#
# @app.route('/doctors')
# def doctor_selection():
#     return render_template('doctor_selection.html')
#
#
# @app.route('/voice-call')
# @app.route('/voice-call/<doctor_id>')
# def voice_call(doctor_id=None):
#     return render_template('voice_call.html')
#
#
# @app.route('/video-call')
# @app.route('/video-call/<doctor_id>')
# def video_call(doctor_id=None):
#     return render_template('video_call.html')
#
#
# @app.route('/health-info')
# def health_info():
#     return render_template('health_info.html')
#
#
# @app.route('/test-calls')
# def test_calls():
#     return render_template('test_calls.html')
#
#
# if __name__ == '__main__':
#     if USE_ML:
#         logger.info("Starting combined ML + rule-based PCOS/PCOD Health Application...")
#     else:
#         logger.info("Starting rule-based PCOS/PCOD Health Application...")
#     app.run(debug=True, host='0.0.0.0', port=5000)




# import os
# import uuid
# import json
# import logging
# from datetime import datetime
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
#
# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key'
# app.permanent_session_lifetime = 1800  # 30 minutes
# CORS(app)
#
# ASSESSMENTS_FILE = 'assessments.csv'
#
# # Define QUESTIONS for the chatbot, aligned with dataset features
# QUESTIONS = [
#     {
#         'id': 'age',
#         'question': 'What is your age range?',
#         'options': ['18-25', '26-30', '31-35', '36-40', '40+']
#     },
#     {
#         'id': 'bmi',
#         'question': 'What is your BMI range?',
#         'options': ['Below 18.5 (Underweight)', '18.5-24.9 (Normal)', '25.0-29.9 (Overweight)', '30.0-34.9 (Obese Class 1)', '35+ (Obese Class 2+)']
#     },
#     {
#         'id': 'cycle_length',
#         'question': 'What is the average length of your menstrual cycle?',
#         'options': ['Less than 21 days', '21-28 days', '29-35 days', 'More than 35 days', 'Irregular/Unpredictable']
#     },
#     {
#         'id': 'menses_length',
#         'question': 'How long do your periods typically last?',
#         'options': ['1-2 days', '3-5 days', '6-7 days', 'More than 7 days']
#     },
#     {
#         'id': 'unusual_bleeding',
#         'question': 'Do you experience abnormal bleeding between periods?',
#         'options': ['Yes', 'No']
#     },
#     {
#         'id': 'cycle_regularity',
#         'question': 'How regular are your menstrual cycles?',
#         'options': ['Very regular (same length each month)', 'Mostly regular (1-2 days variation)', 'Somewhat irregular', 'Very irregular']
#     },
#     {
#         'id': 'symptoms',
#         'question': 'Do you experience symptoms like pelvic pain or cramps?',
#         'options': ['None', 'Occasional (mild)', 'Frequent (moderate)', 'Severe (affects daily life)']
#     },
#     {
#         'id': 'hormonal_issues',
#         'question': 'Do you have signs of hormonal issues (e.g., excessive hair growth, acne, or medical conditions like diabetes or thyroid issues)?',
#         'options': ['None', 'Mild (e.g., occasional acne)', 'Moderate (e.g., noticeable hair growth)', 'Severe (e.g., diagnosed diabetes or thyroid issues)']
#     }
# ]
#
# # Utility: calculate intensity label from symptom/hormonal scores
# def get_intensity_label(symptom_score, hormonal_score):
#     total = symptom_score + hormonal_score
#     if total >= 6:
#         return "High"
#     elif total >= 3:
#         return "Medium"
#     return "Low"
#
# # Convert questionnaire answers to numeric scores used by the heuristics
# def score_answers(answers):
#     # symptom/hormonal scoring
#     symptoms_map = {
#         'None': 0,
#         'Occasional (mild)': 1,
#         'Frequent (moderate)': 2,
#         'Severe (affects daily life)': 3
#     }
#     hormonal_map = {
#         'None': 0,
#         'Mild (e.g., occasional acne)': 1,
#         'Moderate (e.g., noticeable hair growth)': 2,
#         'Severe (e.g., diagnosed diabetes or thyroid issues)': 3
#     }
#     symptom_score = symptoms_map.get(answers.get('symptoms', 'None'), 0)
#     hormonal_score = hormonal_map.get(answers.get('hormonal_issues', 'None'), 0)
#
#     cycle_len = answers.get('cycle_length', '21-28 days')
#     cycle_irregular = cycle_len in ['Irregular/Unpredictable', 'More than 35 days']
#
#     unusual_bleeding = answers.get('unusual_bleeding', 'No') == 'Yes'
#     reg = answers.get('cycle_regularity', 'Very regular (same length each month)')
#     reg_score = 0
#     if reg == 'Very regular (same length each month)':
#         reg_score = 0
#     elif reg == 'Mostly regular (1-2 days variation)':
#         reg_score = 1
#     elif reg == 'Somewhat irregular':
#         reg_score = 2
#     else:
#         reg_score = 3
#
#     bmi_map = {
#         'Below 18.5 (Underweight)': 17.0,
#         '18.5-24.9 (Normal)': 22.0,
#         '25.0-29.9 (Overweight)': 27.0,
#         '30.0-34.9 (Obese Class 1)': 32.0,
#         '35+ (Obese Class 2+)': 37.0
#     }
#     bmi_val = bmi_map.get(answers.get('bmi', '18.5-24.9 (Normal)'), 22.0)
#
#     return {
#         'symptom_score': symptom_score,
#         'hormonal_score': hormonal_score,
#         'cycle_irregular': cycle_irregular,
#         'unusual_bleeding': unusual_bleeding,
#         'reg_score': reg_score,
#         'bmi_val': bmi_val
#     }
#
# # -------- Rule-based predictor ----------
# def rule_based_predict(answers):
#     s = score_answers(answers)
#     symptom = s['symptom_score']
#     hormonal = s['hormonal_score']
#     cycle_irregular = s['cycle_irregular']
#     unusual_bleeding = s['unusual_bleeding']
#     reg_score = s['reg_score']
#     bmi_val = s['bmi_val']
#
#     # default Normal
#     label = 0
#     probs = np.array([0.85, 0.05, 0.05, 0.05])
#
#     # --- Priority: PCOS with PCOD first ---
#     if hormonal >= 2 and cycle_irregular and (unusual_bleeding or symptom >= 2):
#         label = 3
#         probs = np.array([0.02, 0.05, 0.45, 0.48])
#     # --- Then PCOS ---
#     elif hormonal >= 2 and (cycle_irregular or bmi_val >= 30):
#         label = 2
#         probs = np.array([0.05, 0.05, 0.7, 0.2])
#     # --- Then PCOD ---
#     elif (cycle_irregular or reg_score >= 2) and hormonal <= 1 and symptom >= 1:
#         label = 1
#         probs = np.array([0.1, 0.6, 0.15, 0.15])
#     # --- Else stays Normal ---
#
#     # Adjust probabilities slightly
#     total_score = symptom + hormonal + reg_score
#     factor = min(total_score / 10.0, 0.4)
#     if label == 0:
#         probs = probs * (1 - factor) + np.array([1, 0, 0, 0]) * factor
#     elif label == 1:
#         probs = probs * (1 - factor) + np.array([0, 1, 0, 0]) * factor
#     elif label == 2:
#         probs = probs * (1 - factor) + np.array([0, 0, 1, 0]) * factor
#     elif label == 3:
#         probs = probs * (1 - factor) + np.array([0, 0, 0, 1]) * factor
#
#     probs = probs / probs.sum()
#     intensity = get_intensity_label(symptom, hormonal)
#     return int(label), probs.tolist(), intensity
#
#
#
# # Save or update single CSV file with the record for this session
# def save_or_update_record(record_id, answers, label, probs, intensity):
#     """
#     Ensures exactly one CSV file (ASSESSMENTS_FILE) exists.
#     If a record with record_id exists, update it. Otherwise append.
#     """
#     df_row = {
#         'record_id': record_id,
#         'timestamp': datetime.utcnow().isoformat(),
#         'answers_json': json.dumps(answers, ensure_ascii=False),
#         'prediction_label': int(label),
#         'prediction_text': label_to_text(label),
#         'intensity': intensity,
#         'prob_0_normal': float(probs[0]) if len(probs) > 0 else 0.0,
#         'prob_1_pcod': float(probs[1]) if len(probs) > 1 else 0.0,
#         'prob_2_pcos': float(probs[2]) if len(probs) > 2 else 0.0,
#         'prob_3_pcos_pcod': float(probs[3]) if len(probs) > 3 else 0.0
#     }
#     cols = list(df_row.keys())
#     if os.path.exists(ASSESSMENTS_FILE):
#         try:
#             df = pd.read_csv(ASSESSMENTS_FILE)
#             if 'record_id' not in df.columns:
#                 df = pd.DataFrame(columns=cols)
#             if record_id in df['record_id'].astype(str).values:
#                 df.loc[df['record_id'].astype(str) == str(record_id), :] = pd.Series(df_row)
#             else:
#                 df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
#             df.to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessment saved/updated to existing file {ASSESSMENTS_FILE}")
#         except Exception as e:
#             logger.error(f"Error updating assessments file: {e}", exc_info=True)
#             try:
#                 pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#                 logger.info(f"Assessments file overwritten due to error; new file {ASSESSMENTS_FILE} created.")
#             except Exception as e2:
#                 logger.error(f"Failed to write assessments file: {e2}", exc_info=True)
#     else:
#         try:
#             pd.DataFrame([df_row])[cols].to_csv(ASSESSMENTS_FILE, index=False)
#             logger.info(f"Assessments file {ASSESSMENTS_FILE} created and record saved.")
#         except Exception as e:
#             logger.error(f"Failed to create assessments file: {e}", exc_info=True)
#
# def label_to_text(label):
#     return {
#         0: "Normal",
#         1: "PCOD only",
#         2: "PCOS only",
#         3: "PCOS with PCOD"
#     }.get(int(label), "Normal")
#
# # --------- Flask routes ----------
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/chatbot')
# def chatbot():
#     session.clear()
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     logger.debug("Chatbot session initialized (rule-based version)")
#     return render_template('chatbot_pcod.html')
#
# @app.route('/start_chat', methods=['POST'])
# def start_chat():
#     session.permanent = True
#     session['current_question'] = 0
#     session['answers'] = {}
#     session.modified = True
#     response = {
#         'question': QUESTIONS[0]['question'],
#         'options': QUESTIONS[0]['options'],
#         'question_id': QUESTIONS[0]['id']
#     }
#     return jsonify(response)
#
# @app.route('/answer', methods=['POST'])
# def answer():
#     try:
#         data = request.get_json()
#         if not data:
#             logger.error("No JSON data provided in /answer")
#             return jsonify({'error': 'No JSON data provided'}), 400
#         question_id = data.get('question_id')
#         answer_val = data.get('answer')
#         if not question_id or (answer_val is None):
#             logger.error(f"Missing question_id or answer: {data}")
#             return jsonify({'error': 'Missing question_id or answer'}), 400
#         if 'answers' not in session:
#             session['answers'] = {}
#         session['answers'][question_id] = answer_val
#         session['current_question'] = session.get('current_question', 0) + 1
#         session.modified = True
#         logger.debug(f"Answer recorded: {question_id} = {answer_val}, Current question: {session['current_question']}")
#
#         if session['current_question'] < len(QUESTIONS):
#             next_question = QUESTIONS[session['current_question']]
#             response = {
#                 'question': next_question['question'],
#                 'options': next_question['options'],
#                 'question_id': next_question['id']
#             }
#             return jsonify(response)
#         else:
#             answers = session.get('answers', {})
#             label, probs, intensity = rule_based_predict(answers)
#             if 'record_id' not in session:
#                 session['record_id'] = str(uuid.uuid4())
#             record_id = session['record_id']
#             try:
#                 save_or_update_record(record_id, answers, label, probs, intensity)
#             except Exception as e:
#                 logger.error(f"Error saving/updating record: {e}", exc_info=True)
#
#             session['prediction'] = int(label)
#             session['probability'] = probs
#             session['intensity'] = intensity
#             session['condition_type'] = {
#                 'condition': label_to_text(label),
#                 'intensity': intensity,
#                 'description': _get_description_for_label(label),
#                 'medical_advice': _get_advice_for_label(label, intensity)
#             }
#             session.modified = True
#             logger.debug(f"Session result set: label={label}, intensity={intensity}")
#             return jsonify({'completed': True, 'redirect': '/result'})
#     except Exception as e:
#         logger.error(f"Error in /answer: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Server error'}), 500
#
# def _get_description_for_label(label):
#     if label == 0:
#         return "No detected disorder. Normal reproductive function likely."
#     elif label == 1:
#         return "Likely PCOD: predominantly irregular cycles without strong systemic hormonal disturbance."
#     elif label == 2:
#         return "Likely PCOS: systemic hormonal imbalance with features such as hirsutism, metabolic risk, and irregular cycles."
#     elif label == 3:
#         return "Likely combined PCOS with PCOD features: both significant hormonal imbalance and ovarian cystic features."
#     return "Normal reproductive health."
#
# def _get_advice_for_label(label, intensity):
#     if label == 0:
#         return "Maintain a healthy lifestyle. If you have concerns, consult a gynecologist for routine checkups."
#     if label == 1:
#         base = "Consider lifestyle measures (diet, exercise). Consult gynecologist for pelvic ultrasound and cycle management."
#     elif label == 2:
#         base = "See an endocrinologist/gynecologist for hormonal evaluation. Consider metabolic screening (glucose, lipids)."
#     elif label == 3:
#         base = "Seek specialist care (endocrinology + gynecology). Comprehensive hormonal and imaging evaluation recommended."
#     if intensity == "High":
#         return base + " Symptoms severe — seek prompt specialist evaluation."
#     elif intensity == "Medium":
#         return base + " Moderate symptoms — schedule a specialist visit."
#     else:
#         return base + " Mild symptoms — monitor and follow-up if symptoms progress."
#
# @app.route('/result')
# def result():
#     try:
#         prediction = session.get('prediction', 0)
#         probability = session.get('probability', [0.85, 0.05, 0.05, 0.05])
#         condition_type = session.get('condition_type', {
#             'condition': 'Normal',
#             'intensity': 'Low',
#             'description': 'Normal reproductive health',
#             'medical_advice': 'Continue maintaining healthy lifestyle habits.'
#         })
#
#         # color map
#         color_map = {
#             "Normal": "#28a745",
#             "PCOD only": "#ffc107",
#             "PCOS only": "#dc3545",
#             "PCOS with PCOD": "#8b008b"
#         }
#         color = color_map.get(condition_type['condition'], "#28a745")
#
#         return render_template(
#             "results.html",
#             prediction=prediction,
#             probability=probability,
#             condition_type=condition_type['condition'],
#             risk_level=f"{condition_type['intensity']} Risk",
#             severity=condition_type['intensity'],
#             description=condition_type['description'],
#             medical_advice=condition_type['medical_advice'],
#             color=color
#         )
#     except Exception as e:
#         logger.error(f"Error in /result: {str(e)}", exc_info=True)
#         return redirect(url_for('chatbot'))
#
#
# @app.route('/doctors')
# def doctor_selection():
#     return render_template('doctor_selection.html')
#
# @app.route('/voice-call')
# @app.route('/voice-call/<doctor_id>')
# def voice_call(doctor_id=None):
#     return render_template('voice_call.html')
#
# @app.route('/video-call')
# @app.route('/video-call/<doctor_id>')
# def video_call(doctor_id=None):
#     return render_template('video_call.html')
#
# @app.route('/health-info')
# def health_info():
#     return render_template('health_info.html')
#
# @app.route('/test-calls')
# def test_calls():
#     return render_template('test_calls.html')
#
# if __name__ == '__main__':
#     logger.info("Starting rule-based PCOS/PCOD Health Application...")
#     app.run(debug=True, host='0.0.0.0', port=5000)


# import os
# import pickle
# import numpy as np
# from flask import Flask, render_template, request
#
# app = Flask(__name__)
#
# # Load models
# with open("models/pcod_model.pkl", "rb") as f:
#     pcod_model = pickle.load(f)
# with open("models/pcod_scaler.pkl", "rb") as f:
#     pcod_scaler = pickle.load(f)
#
# with open("models/pcos_model.pkl", "rb") as f:
#     pcos_model = pickle.load(f)
# with open("models/pcos_scaler.pkl", "rb") as f:
#     pcos_scaler = pickle.load(f)
#
#
# def get_description_and_advice(prediction, condition):
#     """Map prediction to description and medical advice."""
#     if prediction == "Normal":
#         description = f"Based on your responses, you show minimal indicators of {condition}. Your menstrual patterns and symptoms appear to be within normal ranges."
#         medical_advice = "Continue maintaining healthy lifestyle habits and regular check-ups."
#         condition_type = "Normal"
#         risk_level = "Low Risk"
#         severity = "Normal"
#     else:
#         description = f"Likely {condition}: You may have symptoms consistent with {condition}. Please consult a healthcare provider for a thorough evaluation."
#         medical_advice = f"Consult a {'gynecologist' if condition == 'PCOD' else 'endocrinologist/gynecologist'} for further evaluation, including imaging or hormonal tests."
#         condition_type = condition
#         risk_level = "Moderate Risk"
#         severity = "Moderate"
#
#     return {
#         "description": description,
#         "medical_advice": medical_advice,
#         "condition_type": condition_type,
#         "risk_level": risk_level,
#         "severity": severity
#     }
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# @app.route("/chatbot_pcod")
# def chatbot_pcod():
#     return render_template("chatbot_pcod.html")
#
#
# @app.route("/chatbot_pcos")
# def chatbot_pcos():
#     return render_template("chatbot_pcos.html")
#
#
# @app.route("/predict_pcod", methods=["POST"])
# def predict_pcod():
#     try:
#         # PCOD model expects: Age, BMI, IrregularPeriod, WeightGain, HairGrowth, Acne, HairLoss, FamilyHistory, Pain, StressLevel, InsulinResistance
#         features = [
#             int(request.form["age"]),
#             float(request.form["bmi"]),
#             int(request.form["irregular"]),
#             int(request.form["weightgain"]),
#             int(request.form["hairgrowth"]),
#             int(request.form["acne"]),
#             int(request.form["hairloss"]),
#             int(request.form["family"]),
#             int(request.form["pain"]),
#             int(request.form.get("stress", 0)),  # Default to 0 if not provided
#             int(request.form.get("insulin", 0))  # Default to 0 if not provided
#         ]
#
#         X = pcod_scaler.transform([features])
#         prediction = pcod_model.predict(X)[0]
#         result = "PCOD" if prediction == "High" else "Normal"
#
#         result_data = get_description_and_advice(result, "PCOD")
#
#         return render_template(
#             "results.html",
#             result=f"PCOD Prediction: {result}",
#             prediction=0 if result == "Normal" else 1,
#             probability=[0.85, 0.15] if result == "Normal" else [0.15, 0.85],
#             condition_type=result_data["condition_type"],
#             risk_level=result_data["risk_level"],
#             severity=result_data["severity"],
#             description=result_data["description"],
#             medical_advice=result_data["medical_advice"]
#         )
#     except Exception as e:
#         return render_template("results.html", result=f"Error: {str(e)}")
#
#
# @app.route("/predict_pcos", methods=["POST"])
# def predict_pcos():
#     try:
#         # PCOS model expects: Age, BMI, IrregularPeriods, Infertility, Miscarriages, HairGrowth, Acne, HairLoss, FamilyHistory, StressLevel, InsulinResistance
#         features = [
#             int(request.form["age"]),
#             float(request.form["bmi"]),
#             int(request.form["irregular"]),
#             int(request.form["infertility"]),
#             int(request.form["miscarriage"]),
#             int(request.form["hairgrowth"]),
#             int(request.form["acne"]),
#             int(request.form["hairloss"]),
#             int(request.form["family"]),
#             int(request.form.get("stress", 0)),  # Default to 0 if not provided
#             int(request.form.get("insulin", 0))  # Default to 0 if not provided
#         ]
#
#         X = pcos_scaler.transform([features])
#         prediction = pcos_model.predict(X)[0]
#         result = "PCOS" if prediction == "High" else "Normal"
#
#         result_data = get_description_and_advice(result, "PCOS")
#
#         return render_template(
#             "results.html",
#             result=f"PCOS Prediction: {result}",
#             prediction=0 if result == "Normal" else 1,
#             probability=[0.85, 0.15] if result == "Normal" else [0.15, 0.85],
#             condition_type=result_data["condition_type"],
#             risk_level=result_data["risk_level"],
#             severity=result_data["severity"],
#             description=result_data["description"],
#             medical_advice=result_data["medical_advice"]
#         )
#     except Exception as e:
#         return render_template("results.html", result=f"Error: {str(e)}")
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import json
import time
from datetime import datetime
from datetime import datetime
from flask_cors import CORS
import pandas as pd
import logging
from voice_conversation_routes import voice_conversation_bp


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(voice_conversation_bp)
app.secret_key = 'your-secret-key-change-this-in-production'
app.permanent_session_lifetime = 1800
CORS(app)

# Admin credentials (in production, store this in a database)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = generate_password_hash("admin123")  # Change this password


USERS_FILE = 'users.json'
ASSESSMENTS_FILE = 'assessments.json'


try:
    with open('models/pcos_model.pkl', 'rb') as f:
        pcos_model = pickle.load(f)
    with open('models/pcos_scaler.pkl', 'rb') as f:
        pcos_scaler = pickle.load(f)
    with open('models/pcod_model.pkl', 'rb') as f:
        pcod_model = pickle.load(f)
    with open('models/pcod_scaler.pkl', 'rb') as f:
        pcod_scaler = pickle.load(f)
    with open('models/model_accuracies.pkl', 'rb') as f:
        accuracies = pickle.load(f)
    logger.info("Models and accuracies loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models or accuracies: {e}")
    raise


# Add this helper function after loading the models in app.py

def get_model_accuracy(result, assessment_type):
    """Get the appropriate model accuracy based on result and assessment type"""
    try:
        if result == "PCOS_and_PCOD":
            return f"{accuracies['pcos_pcod_accuracy'] * 100:.1f}%"
        elif result == "Normal":
            return f"{accuracies['normal_accuracy'] * 100:.1f}%"
        elif assessment_type.upper() == "PCOD":
            return f"{accuracies['pcod_accuracy'] * 100:.1f}%"
        elif assessment_type.upper() == "PCOS":
            return f"{accuracies['pcos_accuracy'] * 100:.1f}%"
        else:
            return "N/A"
    except Exception as e:
        logger.error(f"Error getting accuracy: {e}")
        return "N/A"




# @app.route("/predict_pcod", methods=["POST"])
# def predict_pcod():
#     if "user" not in session:
#         flash("Please sign in to make predictions", "error")
#         return redirect(url_for("signin"))
#
#     try:
#         username = session.get("user", "Guest")
#         users = load_users()
#         user_email = users.get(username, {}).get("email", "N/A")
#         form_data = {
#             "age": int(request.form["age"]),
#             "bmi": float(request.form["bmi"]),
#             "irregular": int(request.form["irregular"]),
#             "weight_gain": int(request.form["weight_gain"]),
#             "hairgrowth": int(request.form["hairgrowth"]),
#             "acne": int(request.form["acne"]),
#             "hairloss": int(request.form["hairloss"]),
#             "family": int(request.form["family"]),
#             "pain": int(request.form.get("pain", 0))
#         }
#
#         features = [
#             form_data["age"],
#             form_data["bmi"],
#             form_data["irregular"],
#             form_data["weight_gain"],
#             form_data["hairgrowth"],
#             form_data["acne"],
#             form_data["hairloss"],
#             form_data["family"],
#             form_data["pain"]
#         ]
#
#         X = pcod_scaler.transform([features])
#         probabilities = pcod_model.predict_proba(X)[0]  # [P(Normal), P(PCOD)]
#         prediction = 0 if probabilities[0] > 0.5 else 1
#         result = "Normal" if prediction == 0 else "PCOD"
#
#         result_data = get_description_and_advice(result, "PCOD")
#         assessment_data = {
#             "Age": f"{form_data['age']} years",
#             "BMI": f"{form_data['bmi']:.1f}",
#             "Irregular Periods": "Yes" if form_data['irregular'] == 1 else "No",
#             "Weight Gain": "Yes" if form_data['weight_gain'] == 1 else "No",
#             "Excess Hair Growth": "Yes" if form_data['hairgrowth'] == 1 else "No",
#             "Acne/Oily Skin": "Yes" if form_data['acne'] == 1 else "No",
#             "Hair Thinning/Loss": "Yes" if form_data['hairloss'] == 1 else "No",
#             "Family History": f"{form_data['family']} family member(s)",
#             "Pain": "Yes" if form_data['pain'] == 1 else "No"
#         }
#         assessment_record = {
#             "user": username,
#             "user_email": user_email,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "assessment_type": "PCOD",
#             "form_data": form_data,
#             "result": result,
#             "risk_level": result_data["risk_level"],
#             "severity": result_data["severity"],
#             "prediction_raw": str(prediction)
#         }
#         save_assessment(assessment_record)
#         return render_template(
#             "results.html",
#             result=f"PCOD Prediction: {result}",
#             prediction=prediction,
#             probability=probabilities.tolist(),
#             condition_type=result_data["condition_type"],
#             risk_level=result_data["risk_level"],
#             severity=result_data["severity"],
#             description=result_data["description"],
#             medical_advice=result_data["medical_advice"],
#             recommendations=result_data["recommendations"],
#             assessment_type="pcod",
#             username=username,
#             user_email=user_email,
#             assessment_data=assessment_data,
#             model_accuracy=f"{accuracies['pcod_accuracy'] * 100:.1f}%"
#         )
#     except Exception as e:
#         return render_template("results.html", result=f"Error: {str(e)}", assessment_type="pcod")


@app.route("/predict_pcod", methods=["POST"])
def predict_pcod():
    if "user" not in session:
        flash("Please sign in to make predictions", "error")
        return redirect(url_for("signin"))

    try:
        username = session.get("user", "Guest")
        users = load_users()
        user_email = users.get(username, {}).get("email", "N/A")

        form_data = {
            "age": int(request.form["age"]),
            "bmi": float(request.form["bmi"]),
            "irregular": int(request.form["irregular"]),
            "weightgain": int(request.form["weightgain"]),
            "hairgrowth": int(request.form["hairgrowth"]),
            "acne": int(request.form["acne"]),
            "hairloss": int(request.form["hairloss"]),
            "family": int(request.form["family"]),
            "pain": int(request.form["pain"]),
            "stress": int(request.form.get("stress", 0)),
            "insulin": int(request.form.get("insulin", 0))
        }

        features = [
            form_data["age"], form_data["bmi"], form_data["irregular"],
            form_data["weightgain"], form_data["hairgrowth"], form_data["acne"],
            form_data["hairloss"], form_data["family"], form_data["pain"],
            form_data["stress"], form_data["insulin"]
        ]

        X = pcod_scaler.transform([features])
        prediction = pcod_model.predict(X)[0]
        result = "PCOD" if prediction == "High" else "Normal"

        result_data = get_description_and_advice(result, "PCOD")

        assessment_data = {
            "Age": f"{form_data['age']} years",
            "BMI": f"{form_data['bmi']:.1f}",
            "Irregular Periods": "Yes" if form_data['irregular'] == 1 else "No",
            "Weight Gain": "Yes" if form_data['weightgain'] == 1 else "No",
            "Excess Hair Growth": "Yes" if form_data['hairgrowth'] == 1 else "No",
            "Acne": "Yes" if form_data['acne'] == 1 else "No",
            "Hair Loss": "Yes" if form_data['hairloss'] == 1 else "No",
            "Family History": f"{form_data['family']} family member(s)",
            "Pain": "Yes" if form_data['pain'] == 1 else "No"
        }

        assessment_record = {
            "user": username,
            "user_email": user_email,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assessment_type": "PCOD",
            "form_data": form_data,
            "result": result,
            "risk_level": result_data["risk_level"],
            "severity": result_data["severity"],
            "prediction_raw": str(prediction)
        }
        save_assessment(assessment_record)

        # Get appropriate accuracy
        model_accuracy = get_model_accuracy(result, "PCOD")

        return render_template(
            "results.html",
            result=f"PCOD Prediction: {result}",
            prediction=0 if result == "Normal" else 1,
            probability=[0.85, 0.15] if result == "Normal" else [0.15, 0.85],
            condition_type=result_data["condition_type"],
            risk_level=result_data["risk_level"],
            severity=result_data["severity"],
            description=result_data["description"],
            medical_advice=result_data["medical_advice"],
            assessment_type="pcod",
            username=username,
            user_email=user_email,
            assessment_data=assessment_data,
            model_accuracy=model_accuracy
        )
    except Exception as e:
        logger.error(f"Error in predict_pcod: {e}")
        return render_template("results.html", result=f"Error: {str(e)}", assessment_type="pcod")


@app.route("/predict_pcos", methods=["POST"])
def predict_pcos():
    if "user" not in session:
        flash("Please sign in to make predictions", "error")
        return redirect(url_for("signin"))

    try:
        username = session.get("user", "Guest")
        users = load_users()
        user_email = users.get(username, {}).get("email", "N/A")
        form_data = {
            "age": int(request.form["age"]),
            "bmi": float(request.form["bmi"]),
            "irregular": int(request.form["irregular"]),
            "infertility": int(request.form["infertility"]),
            "miscarriage": int(request.form["miscarriage"]),
            "hairgrowth": int(request.form["hairgrowth"]),
            "acne": int(request.form["acne"]),
            "hairloss": int(request.form["hairloss"]),
            "family": int(request.form["family"])
        }
        # Invert infertility (1 = no difficulty, 0 = difficulty)
        form_data["infertility"] = 1 - form_data["infertility"]

        features = [
            form_data["age"],
            form_data["bmi"],
            form_data["irregular"],
            form_data["infertility"],
            form_data["miscarriage"],
            form_data["hairgrowth"],
            form_data["acne"],
            form_data["hairloss"],
            form_data["family"]
        ]

        # Improved rule-based check for PCOS with PCOD
        critical_features = [
            form_data["irregular"],
            form_data["infertility"] == 0,  # Difficulty conceiving
            form_data["miscarriage"] > 0,
            form_data["hairgrowth"],
            form_data["acne"],
            form_data["hairloss"],
            form_data["family"] > 0
        ]
        if sum(f >= 1 for f in critical_features) >= 5 and form_data["bmi"] >= 25:
            result = "PCOS_and_PCOD"
            prediction = 1
            probability = [0.2, 0.8]  # [P(Normal), P(High)]
        else:
            X = pcos_scaler.transform([features])
            probabilities = pcos_model.predict_proba(X)[0]  # [P(Normal), P(High)]
            prediction = 0 if probabilities[0] > 0.5 else 1
            result = "Normal" if prediction == 0 else "PCOS"

        result_data = get_description_and_advice(result, "PCOS")
        assessment_data = {
            "Age": f"{form_data['age']} years",
            "BMI": f"{form_data['bmi']:.1f}",
            "Irregular Periods": "Yes" if form_data['irregular'] == 1 else "No",
            "Infertility": "Yes" if form_data['infertility'] == 0 else "No",
            "Miscarriages": f"{form_data['miscarriage']} time(s)",
            "Excess Hair Growth": "Yes" if form_data['hairgrowth'] == 1 else "No",
            "Acne/Oily Skin": "Yes" if form_data['acne'] == 1 else "No",
            "Hair Thinning/Loss": "Yes" if form_data['hairloss'] == 1 else "No",
            "Family History": f"{form_data['family']} family member(s)"
        }
        assessment_record = {
            "user": username,
            "user_email": user_email,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assessment_type": "PCOS",
            "form_data": form_data,
            "result": result,
            "risk_level": result_data["risk_level"],
            "severity": result_data["severity"],
            "prediction_raw": str(prediction)
        }
        save_assessment(assessment_record)
        return render_template(
            "results.html",
            result=f"PCOS Prediction: {result}",
            prediction=prediction,
            probability=probability if result == "PCOS_and_PCOD" else probabilities.tolist(),
            condition_type=result_data["condition_type"],
            risk_level=result_data["risk_level"],
            severity=result_data["severity"],
            description=result_data["description"],
            medical_advice=result_data["medical_advice"],
            recommendations=result_data["recommendations"],
            assessment_type="pcos",
            username=username,
            user_email=user_email,
            assessment_data=assessment_data,
            model_accuracy=f"{accuracies['pcos_accuracy'] * 100:.1f}%"
        )
    except Exception as e:
        return render_template("results.html", result=f"Error: {str(e)}", assessment_type="pcos")






def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)


def load_assessments():
    if os.path.exists(ASSESSMENTS_FILE):
        with open(ASSESSMENTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_assessment(assessment_data):
    assessments = load_assessments()
    assessment_id = f"assess_{len(assessments) + 1}_{int(time.time())}"
    assessments[assessment_id] = assessment_data
    with open(ASSESSMENTS_FILE, 'w') as f:
        json.dump(assessments, f, indent=2)
    return assessment_id


def get_dashboard_stats():
    """Get statistics for admin dashboard"""
    users = load_users()
    assessments = load_assessments()
    
    total_users = len(users)
    total_assessments = len(assessments)
    
    pcod_count = sum(1 for a in assessments.values() if a.get('assessment_type') == 'PCOD' and a.get('result') != 'Normal')
    pcos_count = sum(1 for a in assessments.values() if a.get('assessment_type') == 'PCOS' and a.get('result') != 'Normal')
    normal_count = sum(1 for a in assessments.values() if a.get('result') == 'Normal')
    
    return {
        'total_users': total_users,
        'total_assessments': total_assessments,
        'pcod_cases': pcod_count,
        'pcos_cases': pcos_count,
        'normal_cases': normal_count
    }


def get_assessment_distribution():
    """Get assessment type distribution for pie chart"""
    assessments = load_assessments()
    
    pcod_count = sum(1 for a in assessments.values() if a.get('assessment_type') == 'PCOD')
    pcos_count = sum(1 for a in assessments.values() if a.get('assessment_type') == 'PCOS')
    
    return {
        'labels': ['PCOD Assessments', 'PCOS Assessments'],
        'data': [pcod_count, pcos_count]
    }


def get_result_distribution():
    """Get result distribution for pie chart"""
    assessments = load_assessments()
    
    pcod_positive = sum(1 for a in assessments.values() if a.get('assessment_type') == 'PCOD' and a.get('result') != 'Normal')
    pcos_positive = sum(1 for a in assessments.values() if a.get('assessment_type') == 'PCOS' and a.get('result') != 'Normal')
    normal = sum(1 for a in assessments.values() if a.get('result') == 'Normal')
    
    return {
        'labels': ['PCOD Positive', 'PCOS Positive', 'Normal'],
        'data': [pcod_positive, pcos_positive, normal]
    }


def get_symptom_frequency():
    """Get symptom frequency for bar chart"""
    assessments = load_assessments()
    
    symptoms = {
        'Irregular Periods': 0,
        'Weight Gain': 0,
        'Hair Growth': 0,
        'Acne': 0,
        'Hair Loss': 0,
        'Pain': 0,
        'Infertility': 0
    }
    
    for assessment in assessments.values():
        form_data = assessment.get('form_data', {})
        if form_data.get('irregular') == 1:
            symptoms['Irregular Periods'] += 1
        if form_data.get('weightgain') == 1:
            symptoms['Weight Gain'] += 1
        if form_data.get('hairgrowth') == 1:
            symptoms['Hair Growth'] += 1
        if form_data.get('acne') == 1:
            symptoms['Acne'] += 1
        if form_data.get('hairloss') == 1:
            symptoms['Hair Loss'] += 1
        if form_data.get('pain') == 1:
            symptoms['Pain'] += 1
        if form_data.get('infertility') == 1:
            symptoms['Infertility'] += 1
    
    return {
        'labels': list(symptoms.keys()),
        'data': list(symptoms.values())
    }


def get_recent_assessments(limit=10):
    """Get recent assessments for table"""
    assessments = load_assessments()
    
    # Convert to list and sort by timestamp
    assessment_list = []
    for aid, data in assessments.items():
        assessment_list.append({
            'id': aid,
            'user': data.get('user', 'Unknown'),
            'timestamp': data.get('timestamp', 'N/A'),
            'type': data.get('assessment_type', 'N/A'),
            'result': data.get('result', 'N/A'),
            'risk_level': data.get('risk_level', 'N/A')
        })
    
    # Sort by timestamp (newest first)
    assessment_list.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return assessment_list[:limit]


def get_user_growth_data():
    """Get user registration data over time"""
    # For now, return sample data since we don't track registration dates
    # In production, you'd track this in users.json
    return {
        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'data': [10, 15, 25, 35, 45, len(load_users())]
    }



# def get_description_and_advice(prediction, condition):
#     """Map prediction to description and medical advice."""
#     if prediction == "Normal":
#         description = f"Based on your responses, you show minimal indicators of {condition}. Your menstrual patterns and symptoms appear to be within normal ranges."
#         medical_advice = "Continue maintaining healthy lifestyle habits and regular check-ups."
#         condition_type = "Normal"
#         risk_level = "Low Risk"
#         severity = "Normal"
#     else:
#         description = f"Likely {condition}: You may have symptoms consistent with {condition}. Please consult a healthcare provider for a thorough evaluation."
#         medical_advice = f"Consult a {'gynecologist' if condition == 'PCOD' else 'endocrinologist/gynecologist'} for further evaluation, including imaging or hormonal tests."
#         condition_type = condition
#         risk_level = "Moderate Risk"
#         severity = "Moderate"
#
#     return {
#         "description": description,
#         "medical_advice": medical_advice,
#         "condition_type": condition_type,
#         "risk_level": risk_level,
#         "severity": severity
#     }


# def get_description_and_advice(result, condition):
#     if result == "Normal":
#         return {
#             "condition_type": f"No {condition} Detected",
#             "risk_level": "Low",
#             "severity": "Normal",
#             "description": f"Based on your responses, you show minimal indicators of {condition}. Your symptoms appear within normal ranges.",
#             "medical_advice": "Continue maintaining healthy lifestyle habits and regular check-ups."
#         }
#     elif result == "PCOS_and_PCOD":
#         return {
#             "condition_type": "PCOS and PCOD",
#             "risk_level": "Very High",
#             "severity": "Severe",
#             "description": "Your responses indicate a high likelihood of both PCOS and PCOD, with significant symptom overlap.",
#             "medical_advice": "Consult an endocrinologist or gynecologist immediately for comprehensive evaluation and management."
#         }
#     else:
#         return {
#             "condition_type": condition,
#             "risk_level": "High",
#             "severity": "High",
#             "description": f"Your responses indicate a high likelihood of {condition}.",
#             "medical_advice": f"Consult a healthcare provider for a thorough {condition} evaluation."
#         }


# def get_description_and_advice(result, assessment_type):
#     if assessment_type == "PCOS":
#         if result == "PCOS_and_PCOD":
#             return {
#                 "condition_type": "PCOS with PCOD",
#                 "risk_level": "Medium PCOS with PCOD Risk",
#                 "severity": "High",
#                 "description": "Your responses indicate a high likelihood of both PCOS and PCOD, characterized by multiple symptoms such as irregular periods, infertility, and high BMI.",
#                 "medical_advice": "Seek immediate consultation with an endocrinologist or gynecologist for a detailed diagnosis, including blood tests, ultrasound, and possibly fertility assessments.",
#                 "recommendations": [
#                     "Adopt a low-glycemic-index diet to manage insulin resistance and weight.",
#                     "Engage in regular aerobic and strength-training exercises to improve metabolic health.",
#                     "Consult a fertility specialist if planning to conceive.",
#                     "Practice stress-reduction techniques to manage high stress levels.",
#                     "Monitor skin and hair symptoms and discuss treatment options with a dermatologist."
#                 ]
#             }
#         elif result == "PCOS":
#             return {
#                 "condition_type": "PCOS",
#                 "risk_level": "Medium PCOS Risk",
#                 "severity": "High",
#                 "description": "Based on your responses, you show indicators of Polycystic Ovary Syndrome (PCOS), such as irregular periods, infertility, or hormonal imbalances.",
#                 "medical_advice": "Consult a healthcare provider for hormonal tests, ultrasound, and lifestyle management plans to address PCOS symptoms.",
#                 "recommendations": [
#                     "Follow a balanced diet to manage weight and insulin levels.",
#                     "Incorporate regular exercise (e.g., 150 minutes per week) to improve hormonal balance.",
#                     "Track menstrual cycles and report changes to your doctor.",
#                     "Consider consulting a nutritionist for a personalized diet plan."
#                 ]
#             }
#         else:  # Normal
#             return {
#                 "condition_type": "Normal",
#                 "risk_level": "Low Risk",
#                 "severity": "Normal",
#                 "description": "Based on your responses, you show minimal indicators of PCOS. Your symptoms appear to be within normal ranges.",
#                 "medical_advice": "Continue maintaining healthy lifestyle habits and schedule regular check-ups with your healthcare provider.",
#                 "recommendations": [
#                     "Continue a healthy diet to maintain optimal weight and hormonal balance.",
#                     "Stay active with regular exercise to support overall health.",
#                     "Keep track of your menstrual cycle for any future changes.",
#                     "Schedule annual gynecological check-ups to monitor reproductive health."
#                 ]
#             }
#     # Add PCOD case if needed, or handle in a separate function
#     return {
#         "condition_type": "Unknown",
#         "risk_level": "Unknown",
#         "severity": "Unknown",
#         "description": "Unable to determine condition based on provided data.",
#         "medical_advice": "Please consult a healthcare provider for a proper assessment.",
#         "recommendations": ["Consult a healthcare provider for further evaluation."]
#     }


# Updated get_description_and_advice for both PCOD and PCOS
def get_description_and_advice(result, assessment_type):
    if assessment_type == "PCOD":
        if result == "PCOD":
            return {
                "condition_type": "PCOD",
                "risk_level": "Medium PCOD Risk",
                "severity": "Moderate",
                "description": "Based on your responses, you show indicators of Polycystic Ovary Disorder (PCOD), such as irregular periods, weight gain, or other symptoms.",
                "medical_advice": "Consult a healthcare provider for a comprehensive evaluation, including hormonal tests and ultrasound, to confirm PCOD and discuss treatment options.",
                "recommendations": [
                    "Maintain a balanced diet rich in whole grains, vegetables, and lean proteins to manage weight.",
                    "Engage in regular physical activity (e.g., 30 minutes most days) to improve insulin sensitivity.",
                    "Monitor menstrual cycles and report irregularities to your doctor.",
                    "Consider stress management techniques like yoga or meditation."
                ]
            }
        else:  # Normal
            return {
                "condition_type": "Normal",
                "risk_level": "Low Risk",
                "severity": "Normal",
                "description": "Based on your responses, you show minimal indicators of PCOD. Your symptoms appear to be within normal ranges.",
                "medical_advice": "Continue maintaining healthy lifestyle habits and schedule regular check-ups with your healthcare provider.",
                "recommendations": [
                    "Continue a healthy diet to maintain optimal weight and hormonal balance.",
                    "Stay active with regular exercise to support overall health.",
                    "Keep track of your menstrual cycle for any future changes.",
                    "Schedule annual gynecological check-ups to monitor reproductive health."
                ]
            }
    elif assessment_type == "PCOS":
        if result == "PCOS_and_PCOD":
            return {
                "condition_type": "PCOS with PCOD",
                "risk_level": "High PCOS with PCOD Risk",
                "severity": "High",
                "description": "Your responses indicate a high likelihood of both PCOS and PCOD, characterized by multiple severe symptoms including irregular periods, infertility, and high BMI. This combination may increase risks for metabolic and reproductive health issues.",
                "medical_advice": "Seek urgent consultation with an endocrinologist or gynecologist for a detailed diagnosis, including blood tests, ultrasound, and fertility assessments. Early intervention is critical.",
                "recommendations": [
                    "Adopt a low-glycemic-index diet to manage insulin resistance and weight.",
                    "Engage in regular aerobic and strength-training exercises to improve metabolic health.",
                    "Consult a fertility specialist if planning to conceive.",
                    "Practice stress-reduction techniques to manage high stress levels.",
                    "Monitor skin, hair, and weight changes; discuss treatment options with a dermatologist or nutritionist.",
                    "Screen for diabetes and cardiovascular risks due to combined PCOS/PCOD impact."
                ]
            }
        elif result == "PCOS":
            return {
                "condition_type": "PCOS",
                "risk_level": "Medium PCOS Risk",
                "severity": "Moderate",
                "description": "Based on your responses, you show indicators of Polycystic Ovary Syndrome (PCOS), such as irregular periods, infertility, or hormonal imbalances.",
                "medical_advice": "Consult a healthcare provider for hormonal tests, ultrasound, and lifestyle management plans to address PCOS symptoms.",
                "recommendations": [
                    "Follow a balanced diet to manage weight and insulin levels.",
                    "Incorporate regular exercise (e.g., 150 minutes per week) to improve hormonal balance.",
                    "Track menstrual cycles and report changes to your doctor.",
                    "Consider consulting a nutritionist for a personalized diet plan."
                ]
            }
        else:  # Normal
            return {
                "condition_type": "Normal",
                "risk_level": "Low Risk",
                "severity": "Normal",
                "description": "Based on your responses, you show minimal indicators of PCOS or PCOD. Your symptoms appear to be within normal ranges.",
                "medical_advice": "Continue maintaining healthy lifestyle habits and schedule regular check-ups with your healthcare provider.",
                "recommendations": [
                    "Continue a healthy diet to maintain optimal weight and hormonal balance.",
                    "Stay active with regular exercise to support overall health.",
                    "Keep track of your menstrual cycle for any future changes.",
                    "Schedule annual gynecological check-ups to monitor reproductive health."
                ]
            }
    return {
        "condition_type": "Unknown",
        "risk_level": "Unknown",
        "severity": "Unknown",
        "description": "Unable to determine condition based on provided data.",
        "medical_advice": "Please consult a healthcare provider for a proper assessment.",
        "recommendations": ["Consult a healthcare provider for further evaluation."]
    }



# ==================== ROUTES ====================

# Landing/Home Page
@app.route("/")
def home():
    return render_template("home.html")


# User Authentication Routes
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not all([username, email, password, confirm_password]):
            flash("All fields are required", "error")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for("signup"))

        users = load_users()

        if username in users:
            flash("Username already exists", "error")
            return redirect(url_for("signup"))

        users[username] = {
            "email": email,
            "password": generate_password_hash(password)
        }
        save_users(users)

        flash("Registration successful! Please sign in.", "success")
        return redirect(url_for("signin"))

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not all([username, password]):
            flash("All fields are required", "error")
            return redirect(url_for("signin"))

        users = load_users()

        if username in users and check_password_hash(users[username]["password"], password):
            session["user"] = username
            session["user_type"] = "user"
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for("signin"))

    return render_template("signin.html")


# Admin Authentication
@app.route("/admin-signin", methods=["GET", "POST"])
def admin_signin():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session["user"] = username
            session["user_type"] = "admin"
            flash(f"Welcome, Admin!", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin credentials", "error")
            return redirect(url_for("admin_signin"))

    return render_template("admin_signin.html")


# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "success")
    return redirect(url_for("home"))


# User Dashboard (Protected)
@app.route("/index")
def index():
    if "user" not in session or session.get("user_type") != "user":
        flash("Please sign in to access this page", "error")
        return redirect(url_for("signin"))
    return render_template("index.html")


# Admin Dashboard (Protected)
@app.route("/admin-dashboard")
def admin_dashboard():
    if "user" not in session or session.get("user_type") != "admin":
        flash("Admin access required", "error")
        return redirect(url_for("admin_signin"))

    # Get all dashboard data
    stats = get_dashboard_stats()
    assessment_dist = get_assessment_distribution()
    result_dist = get_result_distribution()
    symptom_freq = get_symptom_frequency()
    recent_assessments = get_recent_assessments(10)
    user_growth = get_user_growth_data()
    users = load_users()

    return render_template("admin_dashboard.html", 
                         stats=stats,
                         assessment_dist=assessment_dist,
                         result_dist=result_dist,
                         symptom_freq=symptom_freq,
                         recent_assessments=recent_assessments,
                         user_growth=user_growth,
                         users=users)


# Chatbot Routes (Protected)
@app.route("/chatbot_pcod")
def chatbot_pcod():
    if "user" not in session:
        flash("Please sign in to access chatbot", "error")
        return redirect(url_for("signin"))
    return render_template("chatbot_pcod.html")


@app.route("/chatbot_pcos")
def chatbot_pcos():
    if "user" not in session:
        flash("Please sign in to access chatbot", "error")
        return redirect(url_for("signin"))
    return render_template("chatbot_pcos.html")


@app.route("/blog")
def blog():
    if "user" not in session:
        flash("Please sign in to access the blog", "error")
        return redirect(url_for("signin"))
    return render_template("blog.html")


@app.route("/blog/article/<int:article_id>")
def blog_article(article_id):
    if "user" not in session:
        flash("Please sign in to access the blog", "error")
        return redirect(url_for("signin"))
    
    # Article database
    articles = {
        1: {
            "title": "Understanding PCOS Symptoms: A Complete Guide",
            "category": "Symptoms",
            "icon": "fa-stethoscope",
            "read_time": "5 min read",
            "date": "December 2024",
            "content": """
                <h2>What is PCOS?</h2>
                <p>Polycystic Ovary Syndrome (PCOS) is a hormonal disorder affecting women of reproductive age. It's one of the most common endocrine disorders, affecting approximately 1 in 10 women worldwide.</p>
                
                <h3>Common Symptoms</h3>
                <ul>
                    <li><strong>Irregular Periods:</strong> Infrequent, irregular, or prolonged menstrual cycles are the most common sign of PCOS.</li>
                    <li><strong>Excess Androgen:</strong> Elevated levels of male hormones may result in physical signs such as excess facial and body hair (hirsutism), severe acne, and male-pattern baldness.</li>
                    <li><strong>Polycystic Ovaries:</strong> Your ovaries might be enlarged and contain follicles that surround the eggs, causing the ovaries to fail to function regularly.</li>
                    <li><strong>Weight Gain:</strong> Many women with PCOS struggle with weight management and obesity.</li>
                    <li><strong>Skin Issues:</strong> Darkening of skin, particularly along neck creases, in the groin, and underneath breasts.</li>
                </ul>
                
                <h3>Less Common Symptoms</h3>
                <p>Some women may also experience:</p>
                <ul>
                    <li>Thinning hair or hair loss from the scalp</li>
                    <li>Skin tags in the armpits or neck area</li>
                    <li>Mood changes, including depression and anxiety</li>
                    <li>Sleep apnea</li>
                    <li>Headaches</li>
                </ul>
                
                <h3>When to See a Doctor</h3>
                <p>Consult your healthcare provider if you experience:</p>
                <ul>
                    <li>Irregular periods or no periods at all</li>
                    <li>Signs of excess androgens</li>
                    <li>Difficulty getting pregnant</li>
                    <li>Unexplained weight gain</li>
                </ul>
                
                <h3>Diagnosis</h3>
                <p>PCOS is typically diagnosed through a combination of:</p>
                <ul>
                    <li>Medical history and physical examination</li>
                    <li>Blood tests to measure hormone levels</li>
                    <li>Ultrasound to examine your ovaries</li>
                </ul>
                
                <div class="alert alert-info mt-4">
                    <strong>Important:</strong> Early diagnosis and treatment can help control symptoms and prevent long-term complications such as type 2 diabetes and heart disease.
                </div>
            """
        },
        2: {
            "title": "Medical Treatment Options for PCOS",
            "category": "Treatment",
            "icon": "fa-pills",
            "read_time": "7 min read",
            "date": "December 2024",
            "content": """
                <h2>Medical Treatment Approaches</h2>
                <p>PCOS treatment focuses on managing individual concerns such as infertility, hirsutism, acne, or obesity. Treatment may involve lifestyle changes, medications, or both.</p>
                
                <h3>1. Hormonal Birth Control</h3>
                <p>For women who don't want to become pregnant, hormonal birth control can help:</p>
                <ul>
                    <li>Regulate menstrual cycles</li>
                    <li>Reduce excess hair growth and acne</li>
                    <li>Lower the risk of endometrial cancer</li>
                </ul>
                <p><strong>Options include:</strong> Birth control pills, patches, vaginal rings, or hormonal IUDs.</p>
                
                <h3>2. Insulin-Sensitizing Medications</h3>
                <p><strong>Metformin (Glucophage):</strong></p>
                <ul>
                    <li>Improves insulin resistance</li>
                    <li>Helps regulate menstrual cycles</li>
                    <li>May aid in weight loss</li>
                    <li>Can improve cholesterol levels</li>
                </ul>
                
                <h3>3. Anti-Androgen Medications</h3>
                <p><strong>Spironolactone (Aldactone):</strong></p>
                <ul>
                    <li>Blocks the effects of androgens on the skin</li>
                    <li>Reduces excess hair growth and acne</li>
                    <li>Not recommended during pregnancy</li>
                </ul>
                
                <h3>4. Fertility Medications</h3>
                <p>For women trying to conceive:</p>
                <ul>
                    <li><strong>Clomiphene (Clomid):</strong> First-line fertility treatment</li>
                    <li><strong>Letrozole (Femara):</strong> Alternative to clomiphene</li>
                    <li><strong>Gonadotropins:</strong> Injectable hormones for ovulation induction</li>
                    <li><strong>IVF:</strong> In vitro fertilization for more complex cases</li>
                </ul>
                
                <h3>5. Hair Removal Treatments</h3>
                <ul>
                    <li>Eflornithine (Vaniqa) cream to slow facial hair growth</li>
                    <li>Laser hair removal</li>
                    <li>Electrolysis</li>
                </ul>
                
                <h3>Monitoring and Follow-up</h3>
                <p>Regular monitoring is essential:</p>
                <ul>
                    <li>Blood pressure checks</li>
                    <li>Glucose and cholesterol screening</li>
                    <li>Depression and anxiety screening</li>
                    <li>Sleep apnea evaluation</li>
                </ul>
                
                <div class="alert alert-warning mt-4">
                    <strong>Note:</strong> All medications should be prescribed and monitored by a healthcare professional. Never self-medicate.
                </div>
            """
        },
        3: {
            "title": "PCOS Diet: Foods to Eat and Avoid",
            "category": "Nutrition",
            "icon": "fa-apple-alt",
            "read_time": "6 min read",
            "date": "December 2024",
            "content": """
                <h2>Nutrition and PCOS Management</h2>
                <p>Diet plays a crucial role in managing PCOS symptoms. The right foods can help regulate insulin levels, reduce inflammation, and support hormonal balance.</p>
                
                <h3>Foods to Include</h3>
                
                <h4>1. High-Fiber Foods</h4>
                <ul>
                    <li>Whole grains (quinoa, brown rice, oats)</li>
                    <li>Legumes (lentils, chickpeas, beans)</li>
                    <li>Vegetables (broccoli, cauliflower, Brussels sprouts)</li>
                    <li>Fruits (berries, apples, pears)</li>
                </ul>
                
                <h4>2. Lean Proteins</h4>
                <ul>
                    <li>Fish (salmon, mackerel, sardines)</li>
                    <li>Chicken and turkey</li>
                    <li>Eggs</li>
                    <li>Tofu and tempeh</li>
                    <li>Greek yogurt</li>
                </ul>
                
                <h4>3. Healthy Fats</h4>
                <ul>
                    <li>Avocados</li>
                    <li>Nuts and seeds (almonds, walnuts, chia seeds)</li>
                    <li>Olive oil</li>
                    <li>Fatty fish</li>
                </ul>
                
                <h4>4. Anti-Inflammatory Foods</h4>
                <ul>
                    <li>Turmeric and ginger</li>
                    <li>Green leafy vegetables</li>
                    <li>Berries</li>
                    <li>Green tea</li>
                </ul>
                
                <h3>Foods to Limit or Avoid</h3>
                
                <h4>1. Refined Carbohydrates</h4>
                <ul>
                    <li>White bread and pasta</li>
                    <li>Pastries and cakes</li>
                    <li>Sugary cereals</li>
                </ul>
                
                <h4>2. Sugary Foods and Drinks</h4>
                <ul>
                    <li>Sodas and energy drinks</li>
                    <li>Candy and sweets</li>
                    <li>Fruit juices with added sugar</li>
                </ul>
                
                <h4>3. Processed Foods</h4>
                <ul>
                    <li>Fast food</li>
                    <li>Processed meats</li>
                    <li>Packaged snacks</li>
                </ul>
                
                <h4>4. Inflammatory Foods</h4>
                <ul>
                    <li>Trans fats</li>
                    <li>Excessive red meat</li>
                    <li>Fried foods</li>
                </ul>
                
                <h3>Sample Meal Plan</h3>
                <p><strong>Breakfast:</strong> Greek yogurt with berries and chia seeds<br>
                <strong>Lunch:</strong> Grilled chicken salad with mixed greens and olive oil<br>
                <strong>Snack:</strong> Apple slices with almond butter<br>
                <strong>Dinner:</strong> Baked salmon with quinoa and roasted vegetables</p>
                
                <div class="alert alert-success mt-4">
                    <strong>Tip:</strong> Focus on eating regular, balanced meals to maintain stable blood sugar levels throughout the day.
                </div>
            """
        },
        4: {
            "title": "Exercise and PCOS: Best Workouts for Hormonal Balance",
            "category": "Lifestyle",
            "icon": "fa-dumbbell",
            "read_time": "8 min read",
            "date": "December 2024",
            "content": """
                <h2>The Role of Exercise in PCOS Management</h2>
                <p>Regular physical activity is one of the most effective ways to manage PCOS symptoms. Exercise helps improve insulin sensitivity, aids in weight management, and supports hormonal balance.</p>
                
                <h3>Benefits of Exercise for PCOS</h3>
                <ul>
                    <li>Improves insulin sensitivity and glucose metabolism</li>
                    <li>Aids in weight loss and maintenance</li>
                    <li>Reduces cardiovascular disease risk</li>
                    <li>Improves mood and reduces anxiety</li>
                    <li>Helps regulate menstrual cycles</li>
                    <li>Boosts energy levels</li>
                </ul>
                
                <h3>Best Types of Exercise</h3>
                
                <h4>1. Cardiovascular Exercise</h4>
                <p><strong>Recommended:</strong> 150 minutes of moderate-intensity cardio per week</p>
                <ul>
                    <li>Brisk walking</li>
                    <li>Swimming</li>
                    <li>Cycling</li>
                    <li>Dancing</li>
                    <li>Jogging</li>
                </ul>
                
                <h4>2. Strength Training</h4>
                <p><strong>Recommended:</strong> 2-3 sessions per week</p>
                <ul>
                    <li>Weight lifting</li>
                    <li>Resistance bands</li>
                    <li>Bodyweight exercises</li>
                    <li>Pilates</li>
                </ul>
                <p><strong>Benefits:</strong> Builds muscle mass, which improves insulin sensitivity and metabolism.</p>
                
                <h4>3. High-Intensity Interval Training (HIIT)</h4>
                <p><strong>Recommended:</strong> 1-2 sessions per week</p>
                <ul>
                    <li>Short bursts of intense exercise</li>
                    <li>Followed by recovery periods</li>
                    <li>Highly effective for insulin sensitivity</li>
                </ul>
                
                <h4>4. Mind-Body Exercises</h4>
                <ul>
                    <li><strong>Yoga:</strong> Reduces stress, improves flexibility</li>
                    <li><strong>Tai Chi:</strong> Gentle movement, stress reduction</li>
                    <li><strong>Meditation:</strong> Mental health support</li>
                </ul>
                
                <h3>Sample Weekly Exercise Plan</h3>
                <ul>
                    <li><strong>Monday:</strong> 30 min brisk walk + 20 min strength training</li>
                    <li><strong>Tuesday:</strong> 45 min yoga</li>
                    <li><strong>Wednesday:</strong> 30 min cycling + core exercises</li>
                    <li><strong>Thursday:</strong> Rest or gentle stretching</li>
                    <li><strong>Friday:</strong> 20 min HIIT workout</li>
                    <li><strong>Saturday:</strong> 40 min swimming</li>
                    <li><strong>Sunday:</strong> 30 min walk + meditation</li>
                </ul>
                
                <h3>Exercise Tips for PCOS</h3>
                <ul>
                    <li>Start slowly and gradually increase intensity</li>
                    <li>Find activities you enjoy</li>
                    <li>Exercise with a friend for motivation</li>
                    <li>Track your progress</li>
                    <li>Listen to your body and rest when needed</li>
                    <li>Stay hydrated</li>
                    <li>Combine different types of exercise</li>
                </ul>
                
                <div class="alert alert-info mt-4">
                    <strong>Remember:</strong> Consistency is more important than intensity. Even 10-15 minutes of daily movement can make a difference!
                </div>
            """
        },
        5: {
            "title": "Latest Research on PCOS and Fertility",
            "category": "Research",
            "icon": "fa-microscope",
            "read_time": "10 min read",
            "date": "December 2024",
            "content": """
                <h2>Current Research and Breakthroughs</h2>
                <p>Recent scientific studies have provided new insights into PCOS and fertility, offering hope for better treatments and outcomes.</p>
                
                <h3>Key Research Findings</h3>
                
                <h4>1. Genetic Factors</h4>
                <p>Recent studies have identified several genetic variants associated with PCOS:</p>
                <ul>
                    <li>Genes related to insulin resistance</li>
                    <li>Androgen biosynthesis pathways</li>
                    <li>Ovarian function regulators</li>
                </ul>
                
                <h4>2. Gut Microbiome Connection</h4>
                <p>Emerging research shows a link between gut health and PCOS:</p>
                <ul>
                    <li>Altered gut bacteria composition in PCOS patients</li>
                    <li>Probiotics may help improve insulin sensitivity</li>
                    <li>Diet's impact on gut-hormone axis</li>
                </ul>
                
                <h4>3. Inflammation and PCOS</h4>
                <ul>
                    <li>Chronic low-grade inflammation is common in PCOS</li>
                    <li>Anti-inflammatory treatments show promise</li>
                    <li>Lifestyle modifications reduce inflammatory markers</li>
                </ul>
                
                <h3>Fertility Treatment Advances</h3>
                
                <h4>1. Ovulation Induction</h4>
                <p><strong>Letrozole vs. Clomiphene:</strong></p>
                <ul>
                    <li>Studies show letrozole may be more effective</li>
                    <li>Higher live birth rates with letrozole</li>
                    <li>Fewer side effects reported</li>
                </ul>
                
                <h4>2. IVF Success Rates</h4>
                <ul>
                    <li>Improved protocols for PCOS patients</li>
                    <li>Better management of ovarian hyperstimulation</li>
                    <li>Personalized treatment approaches</li>
                </ul>
                
                <h4>3. Natural Cycle IVF</h4>
                <ul>
                    <li>Minimal stimulation protocols</li>
                    <li>Reduced medication use</li>
                    <li>Lower risk of complications</li>
                </ul>
                
                <h3>Lifestyle Intervention Studies</h3>
                
                <h4>Weight Loss Impact</h4>
                <ul>
                    <li>5-10% weight loss can restore ovulation</li>
                    <li>Improved pregnancy rates</li>
                    <li>Better response to fertility treatments</li>
                </ul>
                
                <h4>Exercise Benefits</h4>
                <ul>
                    <li>Regular exercise improves fertility outcomes</li>
                    <li>Combination of cardio and strength training most effective</li>
                    <li>Stress reduction through exercise helps hormonal balance</li>
                </ul>
                
                <h3>Emerging Treatments</h3>
                
                <h4>1. Inositol Supplementation</h4>
                <ul>
                    <li>Myo-inositol and D-chiro-inositol combination</li>
                    <li>Improves insulin sensitivity</li>
                    <li>May improve egg quality</li>
                </ul>
                
                <h4>2. Vitamin D</h4>
                <ul>
                    <li>Deficiency common in PCOS</li>
                    <li>Supplementation may improve fertility</li>
                    <li>Supports overall reproductive health</li>
                </ul>
                
                <h4>3. Acupuncture</h4>
                <ul>
                    <li>May help regulate menstrual cycles</li>
                    <li>Reduces stress and anxiety</li>
                    <li>Complementary to conventional treatments</li>
                </ul>
                
                <h3>Future Directions</h3>
                <ul>
                    <li>Personalized medicine approaches</li>
                    <li>Gene therapy possibilities</li>
                    <li>Novel drug development</li>
                    <li>Artificial intelligence in treatment planning</li>
                </ul>
                
                <div class="alert alert-success mt-4">
                    <strong>Hope for the Future:</strong> Ongoing research continues to improve our understanding and treatment of PCOS-related fertility issues.
                </div>
            """
        },
        6: {
            "title": "Stress Management and PCOS",
            "category": "Lifestyle",
            "icon": "fa-spa",
            "read_time": "6 min read",
            "date": "December 2024",
            "content": """
                <h2>The Stress-PCOS Connection</h2>
                <p>Stress can significantly impact PCOS symptoms by affecting hormone levels, insulin resistance, and overall health. Learning to manage stress is crucial for PCOS management.</p>
                
                <h3>How Stress Affects PCOS</h3>
                <ul>
                    <li><strong>Cortisol Elevation:</strong> Chronic stress increases cortisol, which can worsen insulin resistance</li>
                    <li><strong>Hormonal Imbalance:</strong> Stress hormones can disrupt reproductive hormones</li>
                    <li><strong>Weight Gain:</strong> Stress-induced cortisol promotes abdominal fat storage</li>
                    <li><strong>Inflammation:</strong> Chronic stress increases inflammatory markers</li>
                    <li><strong>Sleep Disruption:</strong> Stress affects sleep quality, worsening PCOS symptoms</li>
                </ul>
                
                <h3>Effective Stress Management Techniques</h3>
                
                <h4>1. Mindfulness and Meditation</h4>
                <ul>
                    <li><strong>Daily Practice:</strong> 10-20 minutes of meditation</li>
                    <li><strong>Mindful Breathing:</strong> Deep breathing exercises</li>
                    <li><strong>Body Scan:</strong> Progressive muscle relaxation</li>
                    <li><strong>Apps:</strong> Headspace, Calm, Insight Timer</li>
                </ul>
                
                <h4>2. Yoga</h4>
                <p><strong>Benefits for PCOS:</strong></p>
                <ul>
                    <li>Reduces cortisol levels</li>
                    <li>Improves insulin sensitivity</li>
                    <li>Enhances mood and mental clarity</li>
                    <li>Supports weight management</li>
                </ul>
                <p><strong>Recommended Styles:</strong> Hatha, Yin, Restorative yoga</p>
                
                <h4>3. Regular Exercise</h4>
                <ul>
                    <li>Releases endorphins (natural mood boosters)</li>
                    <li>Reduces stress hormones</li>
                    <li>Improves sleep quality</li>
                    <li>Boosts self-esteem</li>
                </ul>
                
                <h4>4. Sleep Hygiene</h4>
                <ul>
                    <li>Aim for 7-9 hours per night</li>
                    <li>Maintain consistent sleep schedule</li>
                    <li>Create relaxing bedtime routine</li>
                    <li>Limit screen time before bed</li>
                    <li>Keep bedroom cool and dark</li>
                </ul>
                
                <h4>5. Social Support</h4>
                <ul>
                    <li>Connect with friends and family</li>
                    <li>Join PCOS support groups</li>
                    <li>Share experiences with others</li>
                    <li>Seek professional counseling if needed</li>
                </ul>
                
                <h4>6. Time Management</h4>
                <ul>
                    <li>Prioritize tasks</li>
                    <li>Learn to say no</li>
                    <li>Delegate when possible</li>
                    <li>Schedule breaks and downtime</li>
                </ul>
                
                <h4>7. Hobbies and Recreation</h4>
                <ul>
                    <li>Engage in activities you enjoy</li>
                    <li>Creative outlets (art, music, writing)</li>
                    <li>Spend time in nature</li>
                    <li>Practice gratitude journaling</li>
                </ul>
                
                <h3>Quick Stress-Relief Techniques</h3>
                <ul>
                    <li><strong>4-7-8 Breathing:</strong> Inhale for 4, hold for 7, exhale for 8</li>
                    <li><strong>Progressive Muscle Relaxation:</strong> Tense and release muscle groups</li>
                    <li><strong>Grounding Exercise:</strong> 5-4-3-2-1 sensory technique</li>
                    <li><strong>Quick Walk:</strong> 5-10 minutes of outdoor walking</li>
                </ul>
                
                <h3>When to Seek Professional Help</h3>
                <p>Consider therapy or counseling if you experience:</p>
                <ul>
                    <li>Persistent anxiety or depression</li>
                    <li>Difficulty coping with daily stress</li>
                    <li>Sleep problems lasting more than 2 weeks</li>
                    <li>Thoughts of self-harm</li>
                </ul>
                
                <div class="alert alert-warning mt-4">
                    <strong>Remember:</strong> Managing stress is not a luxury—it's a necessary part of PCOS treatment. Make it a priority!
                </div>
            """
        },
        7: {
            "title": "PCOD vs PCOS: Understanding the Difference",
            "category": "Symptoms",
            "icon": "fa-female",
            "read_time": "5 min read",
            "date": "December 2024",
            "content": """
                <h2>PCOD vs PCOS: What's the Difference?</h2>
                <p>While often used interchangeably, PCOD (Polycystic Ovarian Disease) and PCOS (Polycystic Ovary Syndrome) are actually different conditions with distinct characteristics.</p>
                
                <h3>PCOD (Polycystic Ovarian Disease)</h3>
                
                <h4>Definition</h4>
                <p>PCOD is a condition where the ovaries release immature or partially mature eggs, which eventually turn into cysts.</p>
                
                <h4>Key Characteristics</h4>
                <ul>
                    <li>More common than PCOS</li>
                    <li>Primarily affects the ovaries</li>
                    <li>Less severe hormonal imbalance</li>
                    <li>Often manageable with lifestyle changes</li>
                    <li>Lower risk of serious complications</li>
                </ul>
                
                <h4>Symptoms</h4>
                <ul>
                    <li>Irregular or absent periods</li>
                    <li>Weight gain</li>
                    <li>Abdominal pain</li>
                    <li>Infertility issues</li>
                    <li>Hair loss or thinning</li>
                </ul>
                
                <h3>PCOS (Polycystic Ovary Syndrome)</h3>
                
                <h4>Definition</h4>
                <p>PCOS is a metabolic disorder that affects the entire endocrine system, not just the ovaries.</p>
                
                <h4>Key Characteristics</h4>
                <ul>
                    <li>Less common than PCOD</li>
                    <li>Affects multiple body systems</li>
                    <li>Significant hormonal imbalance</li>
                    <li>Requires comprehensive medical management</li>
                    <li>Higher risk of complications (diabetes, heart disease)</li>
                </ul>
                
                <h4>Symptoms</h4>
                <ul>
                    <li>Irregular or absent periods</li>
                    <li>Excess androgen (male hormones)</li>
                    <li>Severe acne</li>
                    <li>Excessive hair growth (hirsutism)</li>
                    <li>Male-pattern baldness</li>
                    <li>Insulin resistance</li>
                    <li>Weight gain (especially around abdomen)</li>
                    <li>Darkening of skin</li>
                </ul>
                
                <h3>Comparison Table</h3>
                <table class="table table-bordered mt-3">
                    <thead>
                        <tr>
                            <th>Aspect</th>
                            <th>PCOD</th>
                            <th>PCOS</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Prevalence</strong></td>
                            <td>More common</td>
                            <td>Less common</td>
                        </tr>
                        <tr>
                            <td><strong>Severity</strong></td>
                            <td>Less severe</td>
                            <td>More severe</td>
                        </tr>
                        <tr>
                            <td><strong>Hormonal Impact</strong></td>
                            <td>Mild imbalance</td>
                            <td>Significant imbalance</td>
                        </tr>
                        <tr>
                            <td><strong>Fertility</strong></td>
                            <td>Can conceive with treatment</td>
                            <td>More challenging</td>
                        </tr>
                        <tr>
                            <td><strong>Complications</strong></td>
                            <td>Lower risk</td>
                            <td>Higher risk</td>
                        </tr>
                        <tr>
                            <td><strong>Treatment</strong></td>
                            <td>Lifestyle changes often sufficient</td>
                            <td>Requires medical management</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3>Diagnosis</h3>
                
                <h4>PCOD Diagnosis</h4>
                <ul>
                    <li>Ultrasound showing multiple cysts</li>
                    <li>Hormone level tests</li>
                    <li>Medical history</li>
                </ul>
                
                <h4>PCOS Diagnosis (Rotterdam Criteria)</h4>
                <p>At least 2 of the following 3 criteria:</p>
                <ul>
                    <li>Irregular or absent ovulation</li>
                    <li>Elevated androgen levels</li>
                    <li>Polycystic ovaries on ultrasound</li>
                </ul>
                
                <h3>Treatment Approaches</h3>
                
                <h4>PCOD Treatment</h4>
                <ul>
                    <li>Weight management</li>
                    <li>Regular exercise</li>
                    <li>Balanced diet</li>
                    <li>Stress management</li>
                    <li>Minimal medication usually needed</li>
                </ul>
                
                <h4>PCOS Treatment</h4>
                <ul>
                    <li>Comprehensive lifestyle modifications</li>
                    <li>Hormonal medications</li>
                    <li>Insulin-sensitizing drugs</li>
                    <li>Fertility treatments if needed</li>
                    <li>Regular monitoring</li>
                </ul>
                
                <div class="alert alert-info mt-4">
                    <strong>Important:</strong> Both conditions require proper medical diagnosis and management. Consult a healthcare provider for accurate diagnosis and personalized treatment.
                </div>
            """
        },
        8: {
            "title": "Supplements for PCOS Management",
            "category": "Nutrition",
            "icon": "fa-leaf",
            "read_time": "7 min read",
            "date": "December 2024",
            "content": """
                <h2>Evidence-Based Supplements for PCOS</h2>
                <p>While supplements should not replace medical treatment or a healthy lifestyle, certain supplements have shown promise in managing PCOS symptoms.</p>
                
                <h3>1. Inositol</h3>
                
                <h4>Types</h4>
                <ul>
                    <li><strong>Myo-inositol:</strong> Most studied form</li>
                    <li><strong>D-chiro-inositol:</strong> Works synergistically with myo-inositol</li>
                    <li><strong>Combination (40:1 ratio):</strong> Often most effective</li>
                </ul>
                
                <h4>Benefits</h4>
                <ul>
                    <li>Improves insulin sensitivity</li>
                    <li>Helps regulate menstrual cycles</li>
                    <li>May improve egg quality</li>
                    <li>Reduces androgen levels</li>
                    <li>Supports weight management</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>Typical: 2-4 grams of myo-inositol daily</p>
                
                <h3>2. Vitamin D</h3>
                
                <h4>Why It Matters</h4>
                <ul>
                    <li>Deficiency is common in PCOS (67-85% of patients)</li>
                    <li>Plays role in insulin sensitivity</li>
                    <li>Supports reproductive health</li>
                    <li>Important for bone health</li>
                </ul>
                
                <h4>Benefits</h4>
                <ul>
                    <li>May improve insulin resistance</li>
                    <li>Supports regular menstruation</li>
                    <li>Reduces inflammation</li>
                    <li>May improve fertility outcomes</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>1000-4000 IU daily (get levels tested first)</p>
                
                <h3>3. Omega-3 Fatty Acids</h3>
                
                <h4>Sources</h4>
                <ul>
                    <li>Fish oil (EPA and DHA)</li>
                    <li>Algae oil (vegetarian option)</li>
                    <li>Flaxseed oil (ALA)</li>
                </ul>
                
                <h4>Benefits</h4>
                <ul>
                    <li>Reduces inflammation</li>
                    <li>Improves insulin sensitivity</li>
                    <li>Supports cardiovascular health</li>
                    <li>May reduce androgen levels</li>
                    <li>Supports mental health</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>1000-2000 mg of combined EPA/DHA daily</p>
                
                <h3>4. N-Acetyl Cysteine (NAC)</h3>
                
                <h4>Benefits</h4>
                <ul>
                    <li>Powerful antioxidant</li>
                    <li>Improves insulin sensitivity</li>
                    <li>May improve ovulation</li>
                    <li>Supports liver function</li>
                    <li>Reduces oxidative stress</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>600-1800 mg daily</p>
                
                <h3>5. Magnesium</h3>
                
                <h4>Benefits</h4>
                <ul>
                    <li>Improves insulin sensitivity</li>
                    <li>Reduces inflammation</li>
                    <li>Supports sleep quality</li>
                    <li>Helps manage stress</li>
                    <li>Regulates blood sugar</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>300-400 mg daily (magnesium glycinate or citrate)</p>
                
                <h3>6. Chromium</h3>
                
                <h4>Benefits</h4>
                <ul>
                    <li>Enhances insulin action</li>
                    <li>May improve glucose metabolism</li>
                    <li>Supports weight management</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>200-1000 mcg daily (chromium picolinate)</p>
                
                <h3>7. Berberine</h3>
                
                <h4>Benefits</h4>
                <ul>
                    <li>Comparable to metformin for insulin resistance</li>
                    <li>Improves lipid profiles</li>
                    <li>Supports weight loss</li>
                    <li>May improve ovulation</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>500 mg, 2-3 times daily with meals</p>
                
                <h3>8. Spearmint Tea</h3>
                
                <h4>Benefits</h4>
                <ul>
                    <li>May reduce excess hair growth</li>
                    <li>Anti-androgenic properties</li>
                    <li>Natural and safe</li>
                </ul>
                
                <h4>Dosage</h4>
                <p>2 cups daily</p>
                
                <h3>Important Considerations</h3>
                
                <h4>Before Starting Supplements</h4>
                <ul>
                    <li>Consult with your healthcare provider</li>
                    <li>Get relevant blood tests done</li>
                    <li>Check for potential drug interactions</li>
                    <li>Start one supplement at a time</li>
                    <li>Give each supplement 3-6 months to work</li>
                </ul>
                
                <h4>Quality Matters</h4>
                <ul>
                    <li>Choose reputable brands</li>
                    <li>Look for third-party testing</li>
                    <li>Check for certifications (USP, NSF)</li>
                    <li>Avoid unnecessary additives</li>
                </ul>
                
                <h4>Not a Replacement</h4>
                <p>Supplements work best when combined with:</p>
                <ul>
                    <li>Healthy diet</li>
                    <li>Regular exercise</li>
                    <li>Stress management</li>
                    <li>Adequate sleep</li>
                    <li>Medical treatment as prescribed</li>
                </ul>
                
                <div class="alert alert-warning mt-4">
                    <strong>Caution:</strong> Always consult your healthcare provider before starting any supplement regimen, especially if you're pregnant, trying to conceive, or taking medications.
                </div>
            """
        },
        9: {
            "title": "Natural Remedies for PCOS",
            "category": "Treatment",
            "icon": "fa-heartbeat",
            "read_time": "9 min read",
            "date": "December 2024",
            "content": """
                <h2>Holistic Approaches to PCOS Management</h2>
                <p>Natural remedies can complement medical treatment and help manage PCOS symptoms. These approaches focus on supporting the body's natural healing processes.</p>
                
                <h3>1. Herbal Remedies</h3>
                
                <h4>Cinnamon</h4>
                <ul>
                    <li><strong>Benefits:</strong> Improves insulin sensitivity, regulates blood sugar</li>
                    <li><strong>Usage:</strong> 1-2 teaspoons daily in food or tea</li>
                    <li><strong>Type:</strong> Ceylon cinnamon preferred</li>
                </ul>
                
                <h4>Spearmint</h4>
                <ul>
                    <li><strong>Benefits:</strong> Reduces excess hair growth, anti-androgenic</li>
                    <li><strong>Usage:</strong> 2 cups of spearmint tea daily</li>
                    <li><strong>Duration:</strong> At least 30 days for results</li>
                </ul>
                
                <h4>Fenugreek</h4>
                <ul>
                    <li><strong>Benefits:</strong> Improves glucose metabolism, promotes ovulation</li>
                    <li><strong>Usage:</strong> Seeds or supplement form</li>
                    <li><strong>Dosage:</strong> 500-600 mg daily</li>
                </ul>
                
                <h4>Saw Palmetto</h4>
                <ul>
                    <li><strong>Benefits:</strong> Blocks DHT, reduces excess hair growth</li>
                    <li><strong>Usage:</strong> Supplement form</li>
                    <li><strong>Dosage:</strong> 160 mg twice daily</li>
                </ul>
                
                <h4>Vitex (Chasteberry)</h4>
                <ul>
                    <li><strong>Benefits:</strong> Supports hormonal balance, regulates cycles</li>
                    <li><strong>Usage:</strong> Tincture or capsule</li>
                    <li><strong>Dosage:</strong> 400-1000 mg daily</li>
                </ul>
                
                <h3>2. Acupuncture</h3>
                
                <h4>How It Helps</h4>
                <ul>
                    <li>Regulates menstrual cycles</li>
                    <li>Improves blood flow to ovaries</li>
                    <li>Reduces stress and cortisol</li>
                    <li>May improve insulin sensitivity</li>
                    <li>Supports overall hormonal balance</li>
                </ul>
                
                <h4>Treatment Protocol</h4>
                <ul>
                    <li>Weekly sessions for 3-6 months</li>
                    <li>Maintenance sessions monthly</li>
                    <li>Combine with lifestyle changes</li>
                </ul>
                
                <h3>3. Dietary Approaches</h3>
                
                <h4>Anti-Inflammatory Diet</h4>
                <ul>
                    <li>Focus on whole, unprocessed foods</li>
                    <li>Include colorful vegetables and fruits</li>
                    <li>Choose healthy fats (olive oil, avocado)</li>
                    <li>Limit refined carbohydrates and sugar</li>
                    <li>Include anti-inflammatory spices (turmeric, ginger)</li>
                </ul>
                
                <h4>Low-Glycemic Diet</h4>
                <ul>
                    <li>Prevents blood sugar spikes</li>
                    <li>Improves insulin sensitivity</li>
                    <li>Supports weight management</li>
                    <li>Reduces inflammation</li>
                </ul>
                
                <h4>Intermittent Fasting</h4>
                <ul>
                    <li>May improve insulin sensitivity</li>
                    <li>Supports weight loss</li>
                    <li>Promotes cellular repair</li>
                    <li><strong>Caution:</strong> Not suitable for everyone, consult doctor</li>
                </ul>
                
                <h3>4. Mind-Body Practices</h3>
                
                <h4>Yoga</h4>
                <p><strong>Specific Poses for PCOS:</strong></p>
                <ul>
                    <li>Butterfly pose (Baddha Konasana)</li>
                    <li>Cobra pose (Bhujangasana)</li>
                    <li>Boat pose (Navasana)</li>
                    <li>Bridge pose (Setu Bandhasana)</li>
                    <li>Child's pose (Balasana)</li>
                </ul>
                
                <h4>Meditation</h4>
                <ul>
                    <li>Reduces stress hormones</li>
                    <li>Improves emotional well-being</li>
                    <li>Supports hormonal balance</li>
                    <li>Practice 10-20 minutes daily</li>
                </ul>
                
                <h4>Breathing Exercises (Pranayama)</h4>
                <ul>
                    <li>Alternate nostril breathing</li>
                    <li>Deep belly breathing</li>
                    <li>4-7-8 breathing technique</li>
                </ul>
                
                <h3>5. Essential Oils</h3>
                
                <h4>Beneficial Oils</h4>
                <ul>
                    <li><strong>Lavender:</strong> Stress relief, hormone balance</li>
                    <li><strong>Clary Sage:</strong> Hormonal support, menstrual regulation</li>
                    <li><strong>Peppermint:</strong> Reduces excess hair growth</li>
                    <li><strong>Rosemary:</strong> Improves circulation, hair health</li>
                </ul>
                
                <h4>Usage</h4>
                <ul>
                    <li>Diffuse in your space</li>
                    <li>Add to carrier oil for massage</li>
                    <li>Add to bath water</li>
                    <li>Never ingest without professional guidance</li>
                </ul>
                
                <h3>6. Lifestyle Modifications</h3>
                
                <h4>Sleep Optimization</h4>
                <ul>
                    <li>7-9 hours per night</li>
                    <li>Consistent sleep schedule</li>
                    <li>Dark, cool bedroom</li>
                    <li>Limit blue light before bed</li>
                </ul>
                
                <h4>Stress Reduction</h4>
                <ul>
                    <li>Regular relaxation practices</li>
                    <li>Time in nature</li>
                    <li>Social connections</li>
                    <li>Hobbies and creative activities</li>
                </ul>
                
                <h4>Environmental Toxins</h4>
                <ul>
                    <li>Reduce plastic use</li>
                    <li>Choose organic when possible</li>
                    <li>Use natural cleaning products</li>
                    <li>Avoid endocrine disruptors</li>
                </ul>
                
                <h3>7. Castor Oil Packs</h3>
                
                <h4>Benefits</h4>
                <ul>
                    <li>Reduces inflammation</li>
                    <li>Improves circulation to reproductive organs</li>
                    <li>Supports detoxification</li>
                    <li>May reduce ovarian cysts</li>
                </ul>
                
                <h4>How to Use</h4>
                <ul>
                    <li>Apply castor oil to lower abdomen</li>
                    <li>Cover with cloth and heating pad</li>
                    <li>Leave on for 30-60 minutes</li>
                    <li>Use 3-4 times per week</li>
                </ul>
                
                <h3>Integrative Approach</h3>
                <p>For best results, combine multiple natural remedies with:</p>
                <ul>
                    <li>Medical treatment as prescribed</li>
                    <li>Regular healthcare monitoring</li>
                    <li>Healthy diet and exercise</li>
                    <li>Stress management</li>
                    <li>Adequate sleep</li>
                </ul>
                
                <div class="alert alert-danger mt-4">
                    <strong>Important Warning:</strong> Natural remedies are not a substitute for medical care. Always consult your healthcare provider before starting any new treatment, especially if you're pregnant, trying to conceive, or taking medications.
                </div>
            """
        }
    }
    
    article = articles.get(article_id)
    if not article:
        flash("Article not found", "error")
        return redirect(url_for("blog"))
    
    return render_template("blog_article.html", article=article, article_id=article_id)


# Prediction Routes (Protected)
# @app.route("/predict_pcod", methods=["POST"])
# def predict_pcod():
#     if "user" not in session:
#         flash("Please sign in to make predictions", "error")
#         return redirect(url_for("signin"))
#
#     try:
#         # Get user details
#         username = session.get("user", "Guest")
#         users = load_users()
#         user_email = users.get(username, {}).get("email", "N/A")
#
#         # Collect form data
#         form_data = {
#             "age": int(request.form["age"]),
#             "bmi": float(request.form["bmi"]),
#             "irregular": int(request.form["irregular"]),
#             "weightgain": int(request.form["weightgain"]),
#             "hairgrowth": int(request.form["hairgrowth"]),
#             "acne": int(request.form["acne"]),
#             "hairloss": int(request.form["hairloss"]),
#             "family": int(request.form["family"]),
#             "pain": int(request.form["pain"]),
#             "stress": int(request.form.get("stress", 0)),
#             "insulin": int(request.form.get("insulin", 0))
#         }
#
#         # PCOD model expects: Age, BMI, IrregularPeriod, WeightGain, HairGrowth, Acne, HairLoss, FamilyHistory, Pain, StressLevel, InsulinResistance
#         features = [
#             form_data["age"],
#             form_data["bmi"],
#             form_data["irregular"],
#             form_data["weightgain"],
#             form_data["hairgrowth"],
#             form_data["acne"],
#             form_data["hairloss"],
#             form_data["family"],
#             form_data["pain"],
#             form_data["stress"],
#             form_data["insulin"]
#         ]
#
#         X = pcod_scaler.transform([features])
#         prediction = pcod_model.predict(X)[0]
#         result = "PCOD" if prediction == "High" else "Normal"
#
#         result_data = get_description_and_advice(result, "PCOD")
#
#         # Prepare assessment data with questions and answers
#         assessment_data = {
#             "Age": f"{form_data['age']} years",
#             "BMI": f"{form_data['bmi']:.1f}",
#             "Irregular Periods": "Yes" if form_data['irregular'] == 1 else "No",
#             "Weight Gain": "Yes" if form_data['weightgain'] == 1 else "No",
#             "Excess Hair Growth": "Yes" if form_data['hairgrowth'] == 1 else "No",
#             "Acne": "Yes" if form_data['acne'] == 1 else "No",
#             "Hair Loss": "Yes" if form_data['hairloss'] == 1 else "No",
#             "Family History": f"{form_data['family']} family member(s)",
#             "Pain": "Yes" if form_data['pain'] == 1 else "No"
#         }
#
#         # Save assessment to database
#         assessment_record = {
#             "user": username,
#             "user_email": user_email,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "assessment_type": "PCOD",
#             "form_data": form_data,
#             "result": result,
#             "risk_level": result_data["risk_level"],
#             "severity": result_data["severity"],
#             "prediction_raw": str(prediction)
#         }
#         save_assessment(assessment_record)
#
#         return render_template(
#             "results.html",
#             result=f"PCOD Prediction: {result}",
#             prediction=0 if result == "Normal" else 1,
#             probability=[0.85, 0.15] if result == "Normal" else [0.15, 0.85],
#             condition_type=result_data["condition_type"],
#             risk_level=result_data["risk_level"],
#             severity=result_data["severity"],
#             description=result_data["description"],
#             medical_advice=result_data["medical_advice"],
#             assessment_type="pcod",
#             username=username,
#             user_email=user_email,
#             assessment_data=assessment_data
#         )
#     except Exception as e:
#         return render_template("results.html", result=f"Error: {str(e)}", assessment_type="pcod")


# @app.route("/predict_pcod", methods=["POST"])
# def predict_pcod():
#     if "user" not in session:
#         flash("Please sign in to make predictions", "error")
#         return redirect(url_for("signin"))
#
#     try:
#         username = session.get("user", "Guest")
#         users = load_users()
#         user_email = users.get(username, {}).get("email", "N/A")
#         form_data = {
#             "age": int(request.form["age"]),
#             "bmi": float(request.form["bmi"]),
#             "irregular": int(request.form["irregular"]),
#             "weightgain": int(request.form["weightgain"]),
#             "hairgrowth": int(request.form["hairgrowth"]),
#             "acne": int(request.form["acne"]),
#             "hairloss": int(request.form["hairloss"]),
#             "family": int(request.form["family"]),
#             "pain": int(request.form["pain"]),
#             "stress": int(request.form.get("stress", 0)),
#             "insulin": int(request.form.get("insulin", 0))
#         }
#         features = [
#             form_data["age"],
#             form_data["bmi"],
#             form_data["irregular"],
#             form_data["weightgain"],
#             form_data["hairgrowth"],
#             form_data["acne"],
#             form_data["hairloss"],
#             form_data["family"],
#             form_data["pain"],
#             form_data["stress"],
#             form_data["insulin"]
#         ]
#         X = pcod_scaler.transform([features])
#         prediction = pcod_model.predict(X)[0]
#         result = "PCOD" if prediction == "High" else "Normal"
#         result_data = get_description_and_advice(result, "PCOD")
#         assessment_data = {
#             "Age": f"{form_data['age']} years",
#             "BMI": f"{form_data['bmi']:.1f}",
#             "Irregular Periods": "Yes" if form_data['irregular'] == 1 else "No",
#             "Weight Gain": "Yes" if form_data['weightgain'] == 1 else "No",
#             "Excess Hair Growth": "Yes" if form_data['hairgrowth'] == 1 else "No",
#             "Acne": "Yes" if form_data['acne'] == 1 else "No",
#             "Hair Loss": "Yes" if form_data['hairloss'] == 1 else "No",
#             "Family History": f"{form_data['family']} family member(s)",
#             "Pain": "Yes" if form_data['pain'] == 1 else "No"
#         }
#         assessment_record = {
#             "user": username,
#             "user_email": user_email,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "assessment_type": "PCOD",
#             "form_data": form_data,
#             "result": result,
#             "risk_level": result_data["risk_level"],
#             "severity": result_data["severity"],
#             "prediction_raw": str(prediction)
#         }
#         save_assessment(assessment_record)
#         return render_template(
#             "results.html",
#             result=f"PCOD Prediction: {result}",
#             prediction=0 if result == "Normal" else 1,
#             probability=[0.85, 0.15] if result == "Normal" else [0.15, 0.85],
#             condition_type=result_data["condition_type"],
#             risk_level=result_data["risk_level"],
#             severity=result_data["severity"],
#             description=result_data["description"],
#             medical_advice=result_data["medical_advice"],
#             assessment_type="pcod",
#             username=username,
#             user_email=user_email,
#             assessment_data=assessment_data,
#             model_accuracy=f"{accuracies['pcod_accuracy']*100:.1f}%"
#         )
#     except Exception as e:
#         return render_template("results.html", result=f"Error: {str(e)}", assessment_type="pcod")


# @app.route("/predict_pcos", methods=["POST"])
# def predict_pcos():
#     if "user" not in session:
#         flash("Please sign in to make predictions", "error")
#         return redirect(url_for("signin"))
#
#     try:
#         # Get user details
#         username = session.get("user", "Guest")
#         users = load_users()
#         user_email = users.get(username, {}).get("email", "N/A")
#
#         # Collect form data
#         form_data = {
#             "age": int(request.form["age"]),
#             "bmi": float(request.form["bmi"]),
#             "irregular": int(request.form["irregular"]),
#             "infertility": int(request.form["infertility"]),
#             "miscarriage": int(request.form["miscarriage"]),
#             "hairgrowth": int(request.form["hairgrowth"]),
#             "acne": int(request.form["acne"]),
#             "hairloss": int(request.form["hairloss"]),
#             "family": int(request.form["family"]),
#             "stress": int(request.form.get("stress", 0)),
#             "insulin": int(request.form.get("insulin", 0))
#         }
#
#         # PCOS model expects: Age, BMI, IrregularPeriods, Infertility, Miscarriages, HairGrowth, Acne, HairLoss, FamilyHistory, StressLevel, InsulinResistance
#         features = [
#             form_data["age"],
#             form_data["bmi"],
#             form_data["irregular"],
#             form_data["infertility"],
#             form_data["miscarriage"],
#             form_data["hairgrowth"],
#             form_data["acne"],
#             form_data["hairloss"],
#             form_data["family"],
#             form_data["stress"],
#             form_data["insulin"]
#         ]
#
#         X = pcos_scaler.transform([features])
#         prediction = pcos_model.predict(X)[0]
#         result = "PCOS" if prediction == "High" else "Normal"
#
#         result_data = get_description_and_advice(result, "PCOS")
#
#         # Prepare assessment data with questions and answers
#         assessment_data = {
#             "Age": f"{form_data['age']} years",
#             "BMI": f"{form_data['bmi']:.1f}",
#             "Irregular Periods": "Yes" if form_data['irregular'] == 1 else "No",
#             "Infertility": "Yes" if form_data['infertility'] == 1 else "No",
#             "Miscarriages": f"{form_data['miscarriage']} time(s)",
#             "Excess Hair Growth": "Yes" if form_data['hairgrowth'] == 1 else "No",
#             "Acne/Oily Skin": "Yes" if form_data['acne'] == 1 else "No",
#             "Hair Thinning/Loss": "Yes" if form_data['hairloss'] == 1 else "No",
#             "Family History": f"{form_data['family']} family member(s)"
#         }
#
#         # Save assessment to database
#         assessment_record = {
#             "user": username,
#             "user_email": user_email,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "assessment_type": "PCOS",
#             "form_data": form_data,
#             "result": result,
#             "risk_level": result_data["risk_level"],
#             "severity": result_data["severity"],
#             "prediction_raw": str(prediction)
#         }
#         save_assessment(assessment_record)
#
#         return render_template(
#             "results.html",
#             result=f"PCOS Prediction: {result}",
#             prediction=0 if result == "Normal" else 1,
#             probability=[0.85, 0.15] if result == "Normal" else [0.15, 0.85],
#             condition_type=result_data["condition_type"],
#             risk_level=result_data["risk_level"],
#             severity=result_data["severity"],
#             description=result_data["description"],
#             medical_advice=result_data["medical_advice"],
#             assessment_type="pcos",
#             username=username,
#             user_email=user_email,
#             assessment_data=assessment_data
#         )
#     except Exception as e:
#         return render_template("results.html", result=f"Error: {str(e)}", assessment_type="pcos")


# @app.route("/predict_pcos", methods=["POST"])
# def predict_pcos():
#     if "user" not in session:
#         flash("Please sign in to make predictions", "error")
#         return redirect(url_for("signin"))
#
#     try:
#         username = session.get("user", "Guest")
#         users = load_users()
#         user_email = users.get(username, {}).get("email", "N/A")
#         form_data = {
#             "age": int(request.form["age"]),
#             "bmi": float(request.form["bmi"]),
#             "irregular": int(request.form["irregular"]),
#             "infertility": int(request.form["infertility"]),
#             "miscarriage": int(request.form["miscarriage"]),
#             "hairgrowth": int(request.form["hairgrowth"]),
#             "acne": int(request.form["acne"]),
#             "hairloss": int(request.form["hairloss"]),
#             "family": int(request.form["family"]),
#             "stress": int(request.form.get("stress", 0)),
#             "insulin": int(request.form.get("insulin", 0))
#         }
#         # Check for PCOS and PCOD condition
#         critical_features = [
#             form_data["irregular"],
#             form_data["infertility"],
#             form_data["miscarriage"] > 0,
#             form_data["hairgrowth"],
#             form_data["acne"],
#             form_data["hairloss"],
#             form_data["family"] > 0,
#             form_data["stress"] >= 1,
#             form_data["insulin"]
#         ]
#         if all(f >= 1 for f in critical_features) and form_data["bmi"] >= 25:
#             result = "PCOS_and_PCOD"
#             prediction = "High"
#         else:
#             features = [
#                 form_data["age"],
#                 form_data["bmi"],
#                 form_data["irregular"],
#                 form_data["infertility"],
#                 form_data["miscarriage"],
#                 form_data["hairgrowth"],
#                 form_data["acne"],
#                 form_data["hairloss"],
#                 form_data["family"],
#                 form_data["stress"],
#                 form_data["insulin"]
#             ]
#             X = pcos_scaler.transform([features])
#             prediction = pcos_model.predict(X)[0]
#             result = "PCOS" if prediction == "High" else "Normal"
#         result_data = get_description_and_advice(result, "PCOS")
#         assessment_data = {
#             "Age": f"{form_data['age']} years",
#             "BMI": f"{form_data['bmi']:.1f}",
#             "Irregular Periods": "Yes" if form_data['irregular'] == 1 else "No",
#             "Infertility": "Yes" if form_data['infertility'] == 1 else "No",
#             "Miscarriages": f"{form_data['miscarriage']} time(s)",
#             "Excess Hair Growth": "Yes" if form_data['hairgrowth'] == 1 else "No",
#             "Acne/Oily Skin": "Yes" if form_data['acne'] == 1 else "No",
#             "Hair Thinning/Loss": "Yes" if form_data['hairloss'] == 1 else "No",
#             "Family History": f"{form_data['family']} family member(s)"
#         }
#         assessment_record = {
#             "user": username,
#             "user_email": user_email,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "assessment_type": "PCOS",
#             "form_data": form_data,
#             "result": result,
#             "risk_level": result_data["risk_level"],
#             "severity": result_data["severity"],
#             "prediction_raw": str(prediction)
#         }
#         save_assessment(assessment_record)
#         return render_template(
#             "results.html",
#             result=f"PCOS Prediction: {result}",
#             prediction=0 if result == "Normal" else 1,
#             probability=[0.85, 0.15] if result == "Normal" else [0.15, 0.85],
#             condition_type=result_data["condition_type"],
#             risk_level=result_data["risk_level"],
#             severity=result_data["severity"],
#             description=result_data["description"],
#             medical_advice=result_data["medical_advice"],
#             assessment_type="pcos",
#             username=username,
#             user_email=user_email,
#             assessment_data=assessment_data,
#             model_accuracy=f"{accuracies['pcos_accuracy']*100:.1f}%"
#         )
#     except Exception as e:
#         return render_template("results.html", result=f"Error: {str(e)}", assessment_type="pcos")

#
# if __name__ == "__main__":
#     app.run(debug=True)

# Add these routes
@app.route('/voice_chatbot_pcod')
def voice_chatbot_pcod():
    if "user" not in session:
        flash("Please sign in to access chatbot", "error")
        return redirect(url_for("signin"))
    return render_template("voice_conversation_pcod.html")

@app.route('/voice_chatbot_pcos')
def voice_chatbot_pcos():
    if "user" not in session:
        flash("Please sign in to access chatbot", "error")
        return redirect(url_for("signin"))
    return render_template("voice_conversation_pcos.html")

if __name__ == "__main__":
    app.run(debug=True)


