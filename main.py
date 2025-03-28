import os
import openai
import requests
import wikipedia
import re
import json
import time
import base64
import aiohttp
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Union
from transformers import pipeline, CLIPProcessor, CLIPModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from collections import deque
from PIL import Image
import pytz

class UltimateAI:
    def __init__(self):
        """Initialize all components of the AI assistant"""
        self._initialize_nlp_components()
        self._initialize_memory_systems()
        self._initialize_api_configurations()
        self._setup_math_operations()

    def _initialize_nlp_components(self):
        """Initialize NLP models and processors"""
        self.qa_pipeline = pipeline("question-answering")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _initialize_memory_systems(self):
        """Setup memory and context tracking"""
        self.memory = deque(maxlen=20)
        self.context = {
            'user': {},
            'conversation': [],
            'preferences': {}
        }
        self._load_feedback_data()

    def _initialize_api_configurations(self):
        """Configure all API keys and endpoints"""
        self.api_keys = {
            "wolfram": self._get_env_key("WOLFRAM_API_KEY"),
            "openai": self._get_env_key("OPENAI_API_KEY"),
            "google_search": self._get_env_key("GOOGLE_API_KEY"),
            "news_api": self._get_env_key("NEWS_API_KEY"),
            "weather_api": self._get_env_key("WEATHER_API_KEY"),
            "alpha_vantage": self._get_env_key("ALPHA_VANTAGE_KEY"),
            "google_translate": self._get_env_key("GOOGLE_TRANSLATE_KEY"),
            "twitter": self._get_env_key("TWITTER_API_KEY"),
            "spotify": self._get_env_key("SPOTIFY_CLIENT_ID"),
            "spotify_secret": self._get_env_key("SPOTIFY_CLIENT_SECRET"),
            "tmdb": self._get_env_key("TMDB_API_KEY"),
            "reddit": self._get_env_key("REDDIT_API_KEY")
        }

        self.api_endpoints = {
            "wolfram": "https://api.wolframalpha.com/v1/result",
            "google_search": "https://www.googleapis.com/customsearch/v1",
            "news_api": "https://newsapi.org/v2/top-headlines",
            "weather_api": "https://api.openweathermap.org/data/2.5/weather",
            "alpha_vantage": "https://www.alphavantage.co/query",
            "google_translate": "https://translation.googleapis.com/language/translate/v2",
            "twitter": "https://api.twitter.com/2/tweets/search/recent",
            "spotify_token": "https://accounts.spotify.com/api/token",
            "spotify_search": "https://api.spotify.com/v1/search",
            "tmdb": "https://api.themoviedb.org/3",
            "reddit_oauth": "https://www.reddit.com/api/v1/access_token",
            "reddit_search": "https://oauth.reddit.com/search"
        }

        # Spotify token management
        self.spotify_token = None
        self.spotify_token_expiry = None

    def _setup_math_operations(self):
        """Configure math operations mapping"""
        self.math_ops = {
            'plus': '+', 'added to': '+', 'sum of': '+',
            'difference between': '-', 'minus': '-', 'subtracted from': '-',
            'times': '*', 'multiplied by': '*', 'product of': '*',
            'divided by': '/', 'over': '/'
        }

    def _get_env_key(self, key_name: str) -> str:
        """Safely get environment variable with fallback"""
        return os.getenv(key_name, f"YOUR_{key_name}")

    async def process_input(self, user_input: str, image_path: Optional[str] = None) -> str:
        """
        Main entry point for processing user input
        Args:
            user_input: Text input from user
            image_path: Optional path to image file
        Returns:
            Generated response
        """
        start_time = time.time()
        
        try:
            # Store interaction in memory
            if self.memory:
                self.memory[-1]['ai_response'] = None
            
            # Process based on input type
            if image_path:
                response = await self._process_image(image_path, user_input)
            else:
                response = await self._process_text(user_input)
            
            # Store in memory
            self._store_interaction(user_input, response, start_time)
            
            return response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    async def _process_text(self, text: str) -> str:
        """Route text input to appropriate handler"""
        question_type = self._classify_question(text)
        
        handlers = {
            "math": self._solve_math,
            "news": lambda: self._get_news(self._extract_topic(text, "about")),
            "weather": lambda: self._get_weather(self._extract_topic(text, "in")),
            "stock": lambda: self._get_stock_price(text.split()[-1].upper()),
            "translate": lambda: self._handle_translation(text),
            "twitter": lambda: self._get_tweets(self._extract_topic(text, "about")),
            "music": lambda: self._search_spotify(text.replace("play", "").replace("song", "").strip()),
            "movie": lambda: self._get_movie_info(text.replace("movie", "").replace("about", "").strip()),
            "reddit": lambda: self._get_reddit_posts(self._extract_topic(text, "about"))
        }
        
        handler = handlers.get(question_type, self._general_answer)
        return await handler()

    async def _process_image(self, image_path: str, question: Optional[str] = None) -> str:
        """Process image input with optional question"""
        try:
            image = Image.open(image_path)
            inputs = self.clip_processor(
                text=["a photo of"] if not question else [question],
                images=image,
                return_tensors="pt",
                padding=True
            )
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            return f"I'm {probs[0].max().item()*100:.2f}% confident this image contains: {self._get_top_labels(probs)}"
        except Exception as e:
            return f"Error processing image: {str(e)}"

    # API Integration Methods
    async def _get_news(self, topic: str = "technology", country: str = "us") -> str:
        """Fetch news articles using NewsAPI"""
        try:
            params = {
                "apiKey": self.api_keys["news_api"],
                "q": topic,
                "country": country
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_endpoints["news_api"], params=params) as response:
                    data = await response.json()
                    
                    if "articles" in data and data["articles"]:
                        news_items = [
                            f"📰 {article['title']}\n🔗 {article['url']}"
                            for article in data["articles"][:3]
                        ]
                        return "Latest news:\n" + "\n\n".join(news_items)
                    return "No news found on this topic."
        except Exception as e:
            return f"Error fetching news: {str(e)}"

    async def _get_weather(self, location: Optional[str] = None) -> str:
        """Get current weather data"""
        if not location:
            location = "New York"
            
        try:
            params = {
                "q": location,
                "appid": self.api_keys["weather_api"],
                "units": "metric"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_endpoints["weather_api"], params=params) as response:
                    data = await response.json()
                    
                    if "main" in data:
                        weather_info = (
                            f"🌡️ Temperature: {data['main']['temp']}°C\n"
                            f"☁️ Condition: {data['weather'][0]['description'].capitalize()}\n"
                            f"💧 Humidity: {data['main']['humidity']}%\n"
                            f"💨 Wind: {data['wind']['speed']} km/h"
                        )
                        return weather_info
                    return "Weather data not available for this location."
        except Exception as e:
            return f"Error fetching weather: {str(e)}"

    async def _get_stock_price(self, symbol: str = "AAPL") -> str:
        """Fetch stock market data"""
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_keys["alpha_vantage"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_endpoints["alpha_vantage"], params=params) as response:
                    data = await response.json()
                    
                    if "Global Quote" in data and data["Global Quote"]:
                        quote = data["Global Quote"]
                        return (
                            f"📈 {symbol} Stock Info:\n"
                            f"💰 Price: ${quote['05. price']}\n"
                            f"📅 Change: {quote['10. change percent']}"
                        )
                    return "Stock data not available."
        except Exception as e:
            return f"Error fetching stock data: {str(e)}"

    async def _handle_translation(self, text: str) -> str:
        """Handle translation requests"""
        parts = text.split("to")
        if len(parts) < 2:
            return "Please specify both text and target language."
            
        text_to_translate = parts[0].replace("translate", "").strip()
        target_lang = parts[-1].strip()
        return await self._translate_text(text_to_translate, target_lang)

    async def _translate_text(self, text: str, target_lang: str = "es") -> str:
        """Translate text using Google Translate API"""
        try:
            params = {
                "key": self.api_keys["google_translate"],
                "q": text,
                "target": target_lang
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_endpoints["google_translate"], params=params) as response:
                    data = await response.json()
                    if "data" in data and "translations" in data["data"]:
                        return f"🌍 Translation: {data['data']['translations'][0]['translatedText']}"
                    return "Translation failed."
        except Exception as e:
            return f"Error translating text: {str(e)}"

    async def _get_tweets(self, query: str, max_results: int = 5) -> str:
        """Fetch recent tweets about a topic"""
        try:
            headers = {"Authorization": f"Bearer {self.api_keys['twitter']}"}
            params = {
                "query": query,
                "max_results": max_results,
                "tweet.fields": "created_at,public_metrics"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_endpoints["twitter"],
                    headers=headers,
                    params=params
                ) as response:
                    data = await response.json()
                    
                    if "data" in data and data["data"]:
                        tweets = [
                            f"🐦 {tweet['text']}\n"
                            f"🕒 {tweet['created_at']}\n"
                            f"❤️ {tweet['public_metrics']['like_count']} likes"
                            for tweet in data["data"]
                        ]
                        return "Recent tweets:\n" + "\n\n".join(tweets)
                    return "No tweets found on this topic."
        except Exception as e:
            return f"Error fetching tweets: {str(e)}"

    async def _search_spotify(self, query: str) -> str:
        """Search for music on Spotify"""
        try:
            if not self.spotify_token or time.time() > self.spotify_token_expiry:
                await self._refresh_spotify_token()
            
            headers = {"Authorization": f"Bearer {self.spotify_token}"}
            params = {
                "q": query,
                "type": "track",
                "limit": 3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_endpoints["spotify_search"],
                    headers=headers,
                    params=params
                ) as response:
                    data = await response.json()
                    
                    if "tracks" in data and "items" in data["tracks"]:
                        tracks = [
                            f"🎵 {track['name']} by {', '.join(a['name'] for a in track['artists'])}\n"
                            f"🔗 {track['external_urls']['spotify']}"
                            for track in data["tracks"]["items"]
                        ]
                        return "Spotify results:\n" + "\n\n".join(tracks)
                    return "No matching songs found."
        except Exception as e:
            return f"Error searching Spotify: {str(e)}"

    async def _refresh_spotify_token(self) -> None:
        """Refresh Spotify API access token"""
        try:
            auth_string = f"{self.api_keys['spotify']}:{self.api_keys['spotify_secret']}"
            auth_bytes = auth_string.encode("utf-8")
            auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")
            
            headers = {
                "Authorization": f"Basic {auth_base64}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {"grant_type": "client_credentials"}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_endpoints["spotify_token"],
                    headers=headers,
                    data=data
                ) as response:
                    token_data = await response.json()
                    self.spotify_token = token_data["access_token"]
                    self.spotify_token_expiry = time.time() + token_data["expires_in"] - 60
        except Exception as e:
            print(f"Error refreshing Spotify token: {str(e)}")

    async def _get_movie_info(self, query: str) -> str:
        """Get movie information from TMDB"""
        try:
            params = {"api_key": self.api_keys["tmdb"], "query": query}
            
            async with aiohttp.ClientSession() as session:
                # Search for movie
                async with session.get(
                    f"{self.api_endpoints['tmdb']}/search/movie",
                    params=params
                ) as response:
                    search_data = await response.json()
                    
                    if "results" in search_data and search_data["results"]:
                        movie_id = search_data["results"][0]["id"]
                        
                        # Get detailed info
                        async with session.get(
                            f"{self.api_endpoints['tmdb']}/movie/{movie_id}",
                            params={"api_key": self.api_keys["tmdb"]}
                        ) as detail_response:
                            movie_data = await detail_response.json()
                            
                            return (
                                f"🎬 {movie_data['title']} ({movie_data['release_date'][:4]})\n"
                                f"⭐ Rating: {movie_data['vote_average']}/10\n"
                                f"📝 Overview: {movie_data['overview']}\n"
                                f"🌐 https://www.themoviedb.org/movie/{movie_id}"
                            )
                    return "Movie not found in our database."
        except Exception as e:
            return f"Error fetching movie info: {str(e)}"

    async def _get_reddit_posts(self, query: str, limit: int = 3) -> str:
        """Get top Reddit posts about a topic"""
        try:
            auth = aiohttp.BasicAuth(self.api_keys["reddit"], "")
            data = {"grant_type": "client_credentials"}
            
            async with aiohttp.ClientSession() as session:
                # Get OAuth token
                async with session.post(
                    self.api_endpoints["reddit_oauth"],
                    auth=auth,
                    data=data
                ) as auth_response:
                    token_data = await auth_response.json()
                    access_token = token_data["access_token"]
                
                # Search Reddit
                headers = {"Authorization": f"bearer {access_token}"}
                async with session.get(
                    self.api_endpoints["reddit_search"],
                    headers=headers,
                    params={"q": query, "limit": limit, "sort": "top"}
                ) as search_response:
                    search_data = await search_response.json()
                    
                    if "data" in search_data and "children" in search_data["data"]:
                        posts = [
                            f"📌 {post['data']['title']}\n"
                            f"⬆️ {post['data']['ups']} upvotes\n"
                            f"💬 {post['data']['num_comments']} comments\n"
                            f"🔗 https://reddit.com{post['data']['permalink']}"
                            for post in search_data["data"]["children"]
                        ]
                        return "Top Reddit posts:\n" + "\n\n".join(posts)
                    return "No relevant Reddit posts found."
        except Exception as e:
            return f"Error fetching Reddit posts: {str(e)}"

    # Helper Methods
    def _classify_question(self, text: str) -> str:
        """Determine the type of question"""
        text = text.lower()
        
        if any(word in text for word in ["calculate", "solve", "+", "-", "*", "/"]):
            return "math"
        elif any(word in text for word in ["news", "headlines"]):
            return "news"
        elif any(word in text for word in ["weather", "temperature"]):
            return "weather"
        elif any(word in text for word in ["stock", "share price"]):
            return "stock"
        elif "translate" in text:
            return "translate"
        elif any(word in text for word in ["tweet", "twitter"]):
            return "twitter"
        elif any(word in text for word in ["play", "song", "music"]):
            return "music"
        elif any(word in text for word in ["movie", "film"]):
            return "movie"
        elif "reddit" in text:
            return "reddit"
        else:
            return "general"

    def _extract_topic(self, text: str, keyword: str) -> str:
        """Extract topic after a specific keyword"""
        parts = text.split(keyword)
        return parts[-1].strip() if len(parts) > 1 else ""

    def _store_interaction(self, user_input: str, response: str, start_time: float) -> None:
        """Store interaction in memory"""
        self.memory.append({
            'timestamp': datetime.now(pytz.utc).isoformat(),
            'user_input': user_input,
            'ai_response': response,
            'processing_time': time.time() - start_time,
            'context_snapshot': self.context.copy()
        })

    def _get_top_labels(self, probs, k=3) -> List[str]:
        """Get top labels from CLIP model output"""
        return [f"Label {i+1}" for i in probs[0].argsort(descending=True)[:k]]

    def _load_feedback_data(self) -> None:
        """Load feedback data from file"""
        try:
            with open('feedback_data.json', 'r') as f:
                self.feedback_data = json.load(f)
            self._train_feedback_model()
        except FileNotFoundError:
            self.feedback_data = []

    def _train_feedback_model(self) -> None:
        """Train model on feedback data"""
        if len(self.feedback_data) > 1:
            X = [item['query'] for item in self.feedback_data]
            y = [1 if item['feedback'] == 'positive' else 0 for item in self.feedback_data]
            X_vec = self.vectorizer.fit_transform(X)
            self.classifier.fit(X_vec, y)

    async def _general_answer(self, question: str) -> str:
        """Fallback to general knowledge response"""
        try:
            response = await openai.Completion.acreate(
                engine="text-davinci-003",
                prompt=f"Q: {question}\nA:",
                temperature=0.7,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"I couldn't generate a response. Error: {str(e)}"

    async def _solve_math(self, problem: str) -> str:
        """Solve mathematical problems"""
        try:
            # Try WolframAlpha first
            params = {
                "appid": self.api_keys["wolfram"],
                "i": problem
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_endpoints["wolfram"], params=params) as response:
                    if response.status == 200:
                        return await response.text()
                    
            # Fallback to simple math solver
            for phrase, op in self.math_ops.items():
                if phrase in problem:
                    parts = problem.split(phrase)
                    if len(parts) == 2:
                        try:
                            num1 = self._extract_number(parts[0])
                            num2 = self._extract_number(parts[1])
                            result = eval(f"{num1}{op}{num2}")
                            return f"The answer is {result}"
                        except:
                            continue
            
            return "I couldn't solve this math problem."
        except Exception as e:
            return f"Error solving math problem: {str(e)}"

    def _extract_number(self, text: str) -> Union[int, float]:
        """Extract number from text"""
        try:
            return float(text)
        except ValueError:
            nums = re.findall(r'\d+\.?\d*', text)
            return float(nums[0]) if nums else 0

# Example Usage
async def main():
    ai = UltimateAI()
    
    # Test various functionalities
    print(await ai.process_input("What's the weather in Tokyo?"))
    print(await ai.process_input("Latest news about renewable energy"))
    print(await ai.process_input("Stock price for Tesla"))
    print(await ai.process_input("Translate 'good morning' to Spanish"))
    print(await ai.process_input("Show me tweets about AI advancements"))
    print(await ai.process_input("Play song Bohemian Rhapsody"))
    print(await ai.process_input("Information about the movie Interstellar"))
    print(await ai.process_input("Show me Reddit posts about Python programming"))
    print(await ai.process_input("Calculate 45 divided by 9"))

if __name__ == "__main__":
    asyncio.run(main())