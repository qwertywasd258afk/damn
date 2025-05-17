from flask import Flask, request, jsonify
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='')
co = cohere.Client(os.getenv('COHERE_API_KEY'))


@app.route('/')
def home():

    return app.send_static_file('index.html')


@app.route('/generate-story', methods=['POST'])
def generate_story():
    data = request.json
    theme = data.get('theme')
    setting = data.get('setting')
    plot = data.get('plot')
    characters = data.get('characters')
    length = data.get('length')

    prompt = f"""Write a {length} story with the following elements:
    Theme: {theme}
    Setting: {setting}
    Plot: {plot}
    Characters: {characters}
    
    Make it engaging and creative while maintaining a coherent narrative structure."""

    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=2000,
        temperature=0.8,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )

    return jsonify({'story': response.generations[0].text})


if __name__ == '__main__':
    app.run(debug=True)
