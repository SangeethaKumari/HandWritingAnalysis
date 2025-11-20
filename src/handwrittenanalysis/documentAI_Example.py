from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")

question = "How is the speech"
image = Image.open("/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/english_handwritten.jpg")

pipe(image=image, question=question)

## [{'answer': '20,000$'}]
