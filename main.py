from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", device_map="auto")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")