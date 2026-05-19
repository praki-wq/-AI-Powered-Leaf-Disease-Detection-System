import os
import json
import logging
import sys
import base64

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

from groq import Groq
from dotenv import load_dotenv
from PIL import Image, ImageEnhance


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class DiseaseAnalysisResult:

    disease_detected: bool
    disease_name: Optional[str]
    disease_type: str
    severity: str
    confidence: float
    symptoms: List[str]
    possible_causes: List[str]
    treatment: List[str]
    analysis_timestamp: str = datetime.now().astimezone().isoformat()


class LeafDiseaseDetector:

    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

    DEFAULT_TEMPERATURE = 0.1

    DEFAULT_MAX_TOKENS = 700

    def __init__(self, api_key: Optional[str] = None):

        load_dotenv()

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables"
            )

        self.client = Groq(api_key=self.api_key)

        logger.info("Leaf Disease Detector initialized")

    def create_analysis_prompt(self) -> str:

        return """
You are an expert agricultural AI plant pathologist.

Analyze the uploaded plant leaf image carefully.

STEP 1:
First determine if the image contains a real plant leaf.

If the image contains:
- humans
- animals
- buildings
- random objects
- blurry unclear images
- non-leaf images

then return invalid_image response.

STEP 2:
If this is a valid leaf image, analyze carefully for diseases.

Return detailed JSON response with:

1. disease_name
2. disease_type
3. severity
4. confidence
5. minimum 6 symptoms
6. minimum 6 treatment recommendations
7. minimum 4 possible causes
8. prevention tips

VERY IMPORTANT:
- Compare multiple diseases carefully.
- Do NOT repeatedly predict leaf spot.
- Detect healthy leaves correctly.
- Focus on lesion shape, rust patterns, yellowing, mold, curling, discoloration, and texture.

Supported diseases include:
- Rust Leaf
- Mosaic Virus
- Leaf Spot
- Fungus
- Healthy Leaf
- Powdery Mildew
- Anthracnose

For NON-LEAF images return:

{
    "disease_detected": false,
    "disease_name": null,
    "disease_type": "invalid_image",
    "severity": "none",
    "confidence": 95,
    "symptoms": ["Image does not contain a plant leaf"],
    "possible_causes": ["Invalid image uploaded"],
    "treatment": ["Upload a clear plant leaf image"],
    "prevention_tips": ["Use proper leaf images only"]
}

For VALID LEAF images return ONLY valid JSON:

{
    "disease_detected": true,
    "disease_name": "Exact disease name",
    "disease_type": "fungal/bacterial/viral/healthy",
    "severity": "mild/moderate/severe",
    "confidence": 95,
    "symptoms": [
        "symptom1",
        "symptom2",
        "symptom3",
        "symptom4",
        "symptom5",
        "symptom6"
    ],
    "possible_causes": [
        "cause1",
        "cause2",
        "cause3",
        "cause4"
    ],
    "treatment": [
        "treatment1",
        "treatment2",
        "treatment3",
        "treatment4",
        "treatment5",
        "treatment6"
    ],
    "prevention_tips": [
        "tip1",
        "tip2",
        "tip3",
        "tip4"
    ]
}
Return ONLY valid JSON.
Do not explain anything.
Do not use markdown.
Do not add headings.
Do not add steps.
"""

    def enhance_image(self, base64_image: str):

        image_data = base64.b64decode(base64_image)

        image = Image.open(BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((256, 256))

        contrast = ImageEnhance.Contrast(image)

        image = contrast.enhance(1.5)

        sharpness = ImageEnhance.Sharpness(image)

        image = sharpness.enhance(2.0)

        buffer = BytesIO()

        image.save(buffer, format="JPEG")

        enhanced_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return enhanced_base64

    def analyze_leaf_image_base64(
        self,
        base64_image: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict:

        try:

            logger.info("Starting analysis for base64 image data")

            if not isinstance(base64_image, str):
                raise ValueError("base64_image must be a string")

            if not base64_image:
                raise ValueError("base64_image cannot be empty")

            # Remove data URL prefix if present
            if base64_image.startswith('data:'):
                base64_image = base64_image.split(',', 1)[1]

            # Enhance image before sending to AI
            base64_image = self.enhance_image(base64_image)

            temperature = temperature or self.DEFAULT_TEMPERATURE

            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

            completion = self.client.chat.completions.create(

                model=self.MODEL_NAME,

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.create_analysis_prompt()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],

                temperature=temperature,

                max_completion_tokens=max_tokens,

                top_p=1,

                stream=False,

                stop=None,
            )

            logger.info("API request completed successfully")

            response = completion.choices[0].message.content

            response = response.replace("```json", "")
            response = response.replace("```", "")
            response = response.strip()

            result = self._parse_response(response)

            return result.__dict__

        except Exception as e:

            logger.error(
                f"Analysis failed for base64 image data: {str(e)}"
            )

            raise

    def _parse_response(
        self,
        response_content: str
    ) -> DiseaseAnalysisResult:

        try:

            cleaned_response = response_content.strip()

            if cleaned_response.startswith('```json'):

                cleaned_response = cleaned_response.replace(
                    '```json',
                    ''
                ).replace(
                    '```',
                    ''
                ).strip()

            elif cleaned_response.startswith('```'):

                cleaned_response = cleaned_response.replace(
                    '```',
                    ''
                ).strip()

            disease_data = json.loads(cleaned_response)

            logger.info("Response parsed successfully as JSON")

            return DiseaseAnalysisResult(

                disease_detected=bool(
                    disease_data.get('disease_detected', False)
                ),

                disease_name=disease_data.get('disease_name'),

                disease_type=disease_data.get(
                    'disease_type',
                    'unknown'
                ),

                severity=disease_data.get(
                    'severity',
                    'unknown'
                ),

                confidence=float(
                    disease_data.get('confidence', 0)
                ),

                symptoms=disease_data.get(
                    'symptoms',
                    []
                ),

                possible_causes=disease_data.get(
                    'possible_causes',
                    []
                ),

                treatment=disease_data.get(
                    'treatment',
                    []
                )
            )

        except json.JSONDecodeError:

            logger.warning(
                "Failed to parse JSON, trying regex extraction"
            )

            import re

            json_match = re.search(
                r'\{.*\}',
                response_content,
                re.DOTALL
            )

            if json_match:

                try:

                    disease_data = json.loads(
                        json_match.group()
                    )

                    logger.info(
                        "JSON extracted successfully"
                    )

                    return DiseaseAnalysisResult(

                        disease_detected=bool(
                            disease_data.get(
                                'disease_detected',
                                False
                            )
                        ),

                        disease_name=disease_data.get(
                            'disease_name'
                        ),

                        disease_type=disease_data.get(
                            'disease_type',
                            'unknown'
                        ),

                        severity=disease_data.get(
                            'severity',
                            'unknown'
                        ),

                        confidence=float(
                            disease_data.get(
                                'confidence',
                                0
                            )
                        ),

                        symptoms=disease_data.get(
                            'symptoms',
                            []
                        ),

                        possible_causes=disease_data.get(
                            'possible_causes',
                            []
                        ),

                        treatment=disease_data.get(
                            'treatment',
                            []
                        )
                    )

                except json.JSONDecodeError:
                    pass

            logger.error(
                f"Could not parse response: {response_content}"
            )

            raise ValueError(
                f"Unable to parse API response: {response_content[:200]}..."
            )


def main():

    try:

        detector = LeafDiseaseDetector()

        print(
            "Leaf Disease Detector initialized successfully!"
        )

        print(
            "Use analyze_leaf_image_base64() method."
        )

    except Exception as e:

        print(f"Error: {str(e)}")

        sys.exit(1)


if __name__ == "__main__":
    main()