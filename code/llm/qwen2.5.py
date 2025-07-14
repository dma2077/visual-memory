from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompts[
    0
] = """
你是一个数据格式转化方面的专家，我将会给你一个关于食物图像的烹饪流程以及食物成分的文本，你负责参考下面的描述图像细粒度特征的文本将烹饪流程和食物成分转为关于食物图像的细粒度特征描述的文本，烹饪流程文本为：
Pierce the skin of the chicken with a fork or knife., Sprinkle with kombu tea evenly on both sides of the chicken, about 1 teaspoon per chicken thigh., Brown the skin side of the chicken first over high heat until golden brown., Sprinkle some pepper on the meat just before flipping over., Then brown the other side until golden brown.
食物成分为：
2 Chicken thighs, 2 tsp Kombu tea, 1 White pepper
可参考的图像细粒度特征描述文本为：
In the center of the image, a vibrant blue lunch tray holds four containers, each brimming with a variety of food items. The containers, two in pink and two in yellow, are arranged in a 2x2 grid.\n\nIn the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface.\n\nAdjacent to it in the top right corner, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple. The colors of the apple slices and pineapple chunks contrast beautifully against the pink container.\n\nBelow these, in the bottom left corner of the tray, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets.\n\nFinally, in the bottom right yellow container, there's a sweet treat - a chocolate chip cookie. The golden-brown cookie is dotted with chocolate chips, their dark color standing out against the cookie's lighter surface.\n\nThe arrangement of these containers on the blue tray creates a visually appealing and balanced meal, with each component neatly separated yet part of a cohesive whole.
"""
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
model_path = "/map-vepfs/models/Qwen2.5-72B-Instruct"
llm = LLM(
    model=model_path,
    tensor_parallel_size=8,
    max_seq_len_to_capture=4096
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
