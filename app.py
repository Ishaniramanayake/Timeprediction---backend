from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

app = Flask(__name__)

# Mock model - in a real implementation, you would load a trained model
# For demonstration, we'll create a simple model
def create_mock_model():
    # This would be replaced with your actual trained model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# Initialize encoders and scalers
stroke_encoder = LabelEncoder()
stroke_encoder.fit(['free', 'backstroke', 'fly', 'Individual Medley', 'breastroke'])

distance_encoder = LabelEncoder()
distance_encoder.fit(['50m', '100m', '200m'])

scaler = StandardScaler()

# Load or create model
model = create_mock_model()

@app.route('/api/predict', methods=['POST'])
def predict_time():
    try:
        data = request.json
        
        # Extract features from request
        swimmer_id = data.get('swimmer_id')
        name = data.get('name')
        stroke_type = data.get('stroke_type')
        distance = data.get('distance')
        water_temp = data.get('water_temp', 26.0)  # Default to 26°C if not provided
        humidity = data.get('humidity', 0.7)       # Default to 0.7 if not provided
        prediction_date = data.get('prediction_date')  # Optional date parameter
        
        # Validate required fields
        if not all([swimmer_id, name, stroke_type, distance]):
            return jsonify({
                'error': 'Missing required fields',
                'required': ['swimmer_id', 'name', 'stroke_type', 'distance']
            }), 400
            
        # Process the input data
        # In a real implementation, you would use your trained model to make predictions
        # For this example, we'll use a simple mock prediction
        
        # Encode categorical features
        try:
            encoded_stroke = stroke_encoder.transform([stroke_type])[0]
            encoded_distance = distance_encoder.transform([distance])[0]
        except ValueError:
            return jsonify({
                'error': 'Invalid stroke type or distance',
                'valid_strokes': ['free', 'backstroke', 'fly', 'Individual Medley', 'breastroke'],
                'valid_distances': ['50m', '100m', '200m']
            }), 400
        
        # Create feature array (would be more complex in a real implementation)
        features = np.array([[encoded_stroke, encoded_distance, water_temp, humidity]])
        
        # Mock prediction based on the data
        # In a real implementation, this would use your trained model
        if distance == '50m':
            base_time = 20.5 if stroke_type == 'free' else 23.0
        elif distance == '100m':
            base_time = 46.0 if stroke_type == 'free' else 50.0
        else:  # 200m
            base_time = 102.0 if stroke_type == 'free' else 110.0
            
        # Add some randomness to simulate prediction
        predicted_time = base_time + (np.random.random() - 0.5) * 2
        
        # Format time for response
        if distance == '200m':
            minutes = int(predicted_time // 60)
            seconds = predicted_time % 60
            formatted_time = f"{minutes}:{seconds:.2f}"
        else:
            formatted_time = f"{predicted_time:.2f}"
        
        # Get current date if not provided
        if not prediction_date:
            current_date = datetime.now()
        else:
            try:
                current_date = datetime.strptime(prediction_date, "%Y-%m-%d")
            except ValueError:
                current_date = datetime.now()
        
        # Find upcoming competition based on the date
        upcoming_competition = get_upcoming_competition(current_date)
            
        # Create response
        response = {
            'swimmer_id': swimmer_id,
            'name': name,
            'distance': distance,
            'stroke_type': stroke_type,
            'predicted_best_time': formatted_time,
            'confidence_interval': [formatted_time, f"{float(formatted_time) + 0.5:.2f}"],
            'environmental_factors': {
                'water_temp': water_temp,
                'humidity': humidity
            },
            'prediction_date': current_date.strftime("%Y-%m-%d %H:%M:%S"),
            'competition': upcoming_competition
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_upcoming_competition(date):
    """
    Find the upcoming competition based on the given date
    """
    # Competition calendar for 2025
    competitions = [
        {
            'name': 'French Nationals',
            'start_date': datetime(2025, 4, 8),
            'end_date': datetime(2025, 4, 13),
            'location': 'Tours, France',
            'importance': 'National'
        },
        {
            'name': 'U.S. Nationals',
            'start_date': datetime(2025, 4, 10),
            'end_date': datetime(2025, 4, 14),
            'location': 'Greensboro, USA',
            'importance': 'National'
        },
        {
            'name': 'Aquatics GB Swimming Championships 2025',
            'start_date': datetime(2025, 4, 15),
            'end_date': datetime(2025, 4, 20),
            'location': 'London Aquatics Centre, UK',
            'importance': 'National'
        },
        {
            'name': 'Sri Lanka Aquatic Events - Trial for Junior & World Championships',
            'start_date': datetime(2025, 4, 25),
            'end_date': datetime(2025, 4, 27),
            'location': 'Sugathadasa Stadium, Sri Lanka',
            'importance': 'National'
        },
        {
            'name': 'World Cup #3',
            'start_date': datetime(2025, 5, 1),
            'end_date': datetime(2025, 5, 3),
            'location': 'Markham, Canada',
            'importance': 'International'
        }
    ]
    
    # Find the next competition after the given date
    upcoming = None
    days_until_next = float('inf')
    
    for comp in competitions:
        if comp['start_date'] >= date:
            days_difference = (comp['start_date'] - date).days
            if days_difference < days_until_next:
                days_until_next = days_difference
                upcoming = comp
    
    # If no upcoming competition is found
    if not upcoming:
        return {
            'name': 'No upcoming competitions found',
            'days_until': None
        }
    
    # Format the response
    return {
        'name': upcoming['name'],
        'start_date': upcoming['start_date'].strftime("%Y-%m-%d"),
        'end_date': upcoming['end_date'].strftime("%Y-%m-%d"),
        'location': upcoming['location'],
        'importance': upcoming['importance'],
        'days_until': days_until_next
    }


@app.route('/api/swimmer', methods=['GET'])
def get_swimmer():
    # Mock data based on the Excel file
    swimmers = [
        {"id": "SW001", "name": "Gretchen Claire Walsh", "gender": "Female", "preferred_strokes": ["free", "backstroke", "fly", "Individual Medley"]},
        {"id": "SW002", "name": "Torri Huske", "gender": "Female", "preferred_strokes": ["free", "backstroke", "fly", "Individual Medley", "breaststroke"]},
        {"id": "SW005", "name": "Jordan Crooks", "gender": "Male", "preferred_strokes": ["free", "backstroke", "fly"]},
        {"id": "SW009", "name": "Claire Curzan", "gender": "Female", "preferred_strokes": ["free", "backstroke", "fly"]}
    ]
    
    response = jsonify(swimmers)
    response.headers.add('Content-Type', 'application/json')
    return response

@app.route('/api/history/<swimmer_id>', methods=['GET'])
def get_history(swimmer_id):
    # Mock historical data based on the Excel file
    if swimmer_id == "SW001":
        history = [
            {"competition": "2025 ACC Championships", "date": "2025-02-19", "distance": "50m", "stroke_type": "free", "time": "20.6"},
            {"competition": "2025 ACC Championships", "date": "2025-02-21", "distance": "100m", "stroke_type": "backstroke", "time": "48.95"},
            {"competition": "2025 ACC Championships", "date": "2025-02-22", "distance": "100m", "stroke_type": "free", "time": "45.2"},
            {"competition": "2024 SC World Champs", "date": "2024-12-15", "distance": "50m", "stroke_type": "free", "time": "22.83"}
        ]
    elif swimmer_id == "SW005":
        history = [
            {"competition": "2025 SEC Championships", "date": "2025-02-19", "distance": "50m", "stroke_type": "free", "time": "17.85"},
            {"competition": "2025 SEC Championships", "date": "2025-02-22", "distance": "100m", "stroke_type": "free", "time": "40.45"},
            {"competition": "2024 SC World Champs", "date": "2024-12-14", "distance": "50m", "stroke_type": "free", "time": "19.9"}
        ]

    elif swimmer_id == "SW002":
        history = [
            {"competition": "2025 NCAA Championships", "date": "2025-03-18", "distance": "100m", "stroke_type": "fly", "time": "49.42"},
            {"competition": "2025 NCAA Championships", "date": "2025-03-20", "distance": "50m", "stroke_type": "free", "time": "21.35"},
            {"competition": "2024 SC World Champs", "date": "2024-12-16", "distance": "100m", "stroke_type": "fly", "time": "54.10"}
        ]
    else:
        history = []
    
    return jsonify(history)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
