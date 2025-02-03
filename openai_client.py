import openai
import logging
from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS

logger = logging.getLogger(__name__)
openai.api_key = OPENAI_API_KEY

def get_prediction(analysis_summary):
    """Get prediction from OpenAI API"""
    try:
        prompt = f"""
                You must give a VERY short-term price movement prediction for the next 30 SECONDS.
                Remember that price usually moves insignificantly in 30 seconds!
                
                Current technical indicators:
                - MACD: {analysis_summary['macd_color']}
                - RSI: {analysis_summary['rsi']:.2f}
                - Volume: {analysis_summary['volume_trend']}
                - Short-term momentum (30 sec): {analysis_summary['short_term_momentum']:.2f}
                - Trend strength ADX: {analysis_summary['trend_strength_adx']} ({analysis_summary['adx']:.1f})
                
                Additional indicators:
                - Instant volatility: {analysis_summary['volatility']}
                - Stochastic: K({analysis_summary['stoch_k']:.1f}) D({analysis_summary['stoch_d']:.1f})
                - Current price relative to Bollinger Bands:
                  Upper: {analysis_summary['bb_upper']:.2f}
                  Middle: {analysis_summary['bb_middle']:.2f}
                  Lower: {analysis_summary['bb_lower']:.2f}
                - Average True Range (ATR): {analysis_summary['atr']:.2f}
                
                Consider that:
                1. This is an ultra-short-term 30-second forecast
                2. Price movement in 30 seconds usually ranges from 0.01-0.05%
                3. Short-term momentum and impulse are most important
                
                Give the answer strictly in the format:
                PREDICTION: [UP or DOWN in the next 30 seconds]
                REASON: [brief explanation focusing on short-term indicators]
                CONFIDENCE: [number from 0 to 100]%
                """
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an algorithmic trader specializing in ultra-short-term trading (scalping). Your task is to predict price movement only for the next 30 seconds."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting prediction from OpenAI: {str(e)}")
        return None 