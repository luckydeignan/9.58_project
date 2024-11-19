def get_caption_prompt():
    CAPTION_PROMPTS = [
    "Create a 10-word description of the image. The image is from the Natural Scenes Dataset, so focus your description on the nature and any figures present in the image. Only include your description and no other words in your response."
    "Write a descriptive 10-word caption that captures this image's essence. Only include the caption and no other words in your response.",
    "Create a 10-word caption that highlights the main scene elements. Only include the caption and no other words in your response.",
    "Generate a 10-word caption summarizing the key details in this image. Only include the caption and no other words in your response.",
    "Describe this image in a compelling 10-word caption focusing on nature. Only include the caption and no other words in your response.",
    "Provide a concise 10-word caption capturing the natural scene depicted. Only include the caption and no other words in your response.",
    "Summarize this scene with an insightful 10-word caption. Only include the caption and no other words in your response.",
    "Craft a vivid 10-word caption describing the image's primary features. Only include the caption and no other words in your response.",
    "In 10 words, describe the mood and setting of this scene. Only include the caption and no other words in your response.",
    "Give a 10-word caption that encapsulates the environment shown. Only include the caption and no other words in your response.",
    "Describe the atmosphere of this image in a 10-word caption. Only include the caption and no other words in your response.",
    "Offer a detailed 10-word caption that highlights scene textures. Only include the caption and no other words in your response.",
    "Summarize this image's natural elements with an engaging 10-word caption. Only include the caption and no other words in your response.",
    "Write a 10-word caption focusing on colors and setting. Only include the caption and no other words in your response.",
    "Provide a 10-word caption that encapsulates the visual appeal here. Only include the caption and no other words in your response.",
    "Create a 10-word caption capturing the scene's mood and primary focus. Only include the caption and no other words in your response.",
    "Describe the central theme of this image in 10 words. Only include the caption and no other words in your response.",
    "Compose a 10-word caption that brings out the scene's beauty. Only include the caption and no other words in your response.",
    "In 10 words, highlight the main objects and landscape in this scene. Only include the caption and no other words in your response.",
    "Provide a descriptive 10-word caption with an emphasis on nature. Only include the caption and no other words in your response.",
    "Craft a captivating 10-word caption summarizing the scene's main elements. Only include the caption and no other words in your response."
]
    
    prompts = []

    prompts.append("Give this image a well-described caption in exactly 10 words, \
Focus on the scene and the objects in it, and describe it as if you were explaining it to a 5 year old. \
Only include the caption and no other words in your response")
    prompts.append("Give this image a well-described caption in exactly 10 words, \
Focus on the subject of the image. Only include the caption and no other words in your response")
    prompts.append("Write a descriptive 10-word caption that captures this image’s essence.")

    for cap in CAPTION_PROMPTS:
        prompts.append(cap)

    return prompts

# print(get_caption_prompt())