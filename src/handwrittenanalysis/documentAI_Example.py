from transformers import pipeline
from PIL import Image
image_path_1 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwrittenanalysis/tailscale.jpeg"
image_path_2 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwrittenanalysis/english_handwritten.jpg"
image_path_3 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwrittenanalysis/info.jpg"
image_path_4 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwrittenanalysis/asif_notes.jpeg"
pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")

question = "What questions are available in the document?"
#image = Image.open("/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwrittenanalysis/tailscale.jpeg")

text = pipe(image=image_path_3, question=question)
print("**********")
print(text)

## [{'answer': '20,000$'}]



#ocr = pipeline("image-to-text", model="microsoft/trocr-base-printed")
#text = ocr(image_path_3)[0]["generated_text"]


nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

nlp(
    image_path_4,
    "What is the last name?"
)