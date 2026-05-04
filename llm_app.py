import pandas as pd
from openai import OpenAI
from tqdm import tqdm

def use_llm_function():

    # 定义一个内部的函数
    # 函数输入是字符串，输入的是ABCD上的脑测量描述
    # 输出的是解析后的，字符串
    def connect_deepseek(input_string):

        # 设定提示词
        prompt = '''
        我有一些关于脑科学的数据，脑测量变量。
        但是这一领域门槛比较高，我作为一个外行，不是多么懂，需要你给我解析一下脑测量的变量含义
        
        输出规则如下
        规则：我给你一句话，这句话是，脑测量的变量含义，是英文。你需要给我解析出这句话指代的是什么脑结构，并且告诉我中文含义
        规则：你的输出是，脑结构英文名称（英文输出部分），变量的中文含义（中文输出部分）。之间使用-----符号连接，且-----是唯一的分隔符
        规则：不要输出其他的文字，思考的过程也不要出现在回复内容里
        规则：注意辨别脑区的分割方法等词，脑区的分割方法不应该出现在任何的输出结果内，比如Destrieux ROI就是脑区的分割方法
        规则（英文输出部分）：脑结构英文输出应该使用临近的上位词而不是精确词
            例如superior segment of the circular sulcus of the insula应输出insula一个词。
            例如Average T1-weighted intensity (gray/white contrast) in Destrieux ROI: gyrus rectus (right hemisphere)
            英文部分应该输出为gyrus rectus，不应只输出gyrus，因为只输出gyrus脑回一词，太笼统
        规则（英文输出部分）：英文输出部分，应最终总结为某个脑结构或者某个脑沟、脑回，不应该带有形容、修饰等前缀，不应带有左右半球等词
            但是，如果实在没有具体脑结构，那么可以输出简短的全称
            例如Average sulcal depth in Destrieux ROI: posterior transverse collateral sulcus (right hemisphere)
            可以输出为posterior transverse collateral sulcus，不应只输出为sulcus因为sulcus太过笼统
        规则（中文输出部分）：中文输出部分，要简洁、正式、流畅
        规则（中文输出部分）：中文输出部分，应输出为，半球，脑结构，加测量方式的组合
            例如Average sulcal depth in Destrieux ROI: 
               superior segment of the circular sulcus of the insula (left hemisphere)
            应输出，左半球岛叶环沟上段沟深度
            如果脑测量中，没有写明左右半球，中文输出中可以不说左右半球这一项
        规则（中文输出部分）：中文输出部分，脑测量的计算方式，应笼统的描述
            例如：Average sulcal depth只输出为“沟深度”，不应输出为“平均沟深度”。例如Average cortical thickness只输出“皮质厚度”
            例如：Average T1-weighted intensity只输出为“T1强度”即可，不应输出为“T1加权平均强度”
            但如果英文中表示为“总体积”、“总面积”等，测量方式的中文输出需要带一个“总”字
        规则（中文输出部分）：如果英文原文中出现，灰质、白质、灰白对比等描述，你不要在中文输出中，进行描述
            例如Average T1-weighted intensity (gray matter) in Destrieux ROI: gyrus rectus (left hemisphere)
            中文输出中不应带有灰质、白质、灰白对比等相关描述
        规则（中文输出部分）：如果英文原文中出现“gyri and sulci”即“沟”、“回”同时出现，中文只需解析为“沟回”即可
            不要输出为“脑沟和脑回”
        规则（中文输出部分）：如果英文直译的译文为“后横侧支沟沟深度”这种，连续的两个字“沟沟深度”，将其简化为“后横侧支沟深度”
            例如：“某某脑沟沟深度”，应该解析为“某某脑沟深度”
        
        举例：
        我传给你的：
        Meaning of brain measurement variables in English
        你的回答：
        Brain structure-----中文含义
        
        举例：
        我传给你的：
        Total volume of Destrieux ROI: horizontal ramus of the anterior segment of the lateral sulcus (left hemisphere)
        你的回答：
        lateral sulcus-----左半球外侧沟水平前支总体积
        
        举例：
        我传给你的：
        Total surface area of Destrieux ROI: all (left hemisphere)
        你的回答：
        surface area-----左半球总皮质表面积
        
        举例：
        我传给你的：
        Average T1-weighted intensity in Subcortical ROI: ventral diencephalon (right hemisphere)
        （你应该先判断，ventral diencephalon是一个脑结构，还是说ventral只是一个修饰词，真正的脑结构是diencephalon这个词
        其实diencephalon这个才是真正的脑结构，ventral这个是修饰词，所以根据规则，你应该只保留脑结构的词，去掉修饰词）
        所以，你的输出应该是：
        diencephalon-----右半球腹侧间脑T1强度
        '''

        client = OpenAI(api_key='sk-0c7b71446e6e4c90a29475fd67783eb5',
                        base_url='https://api.deepseek.com')

        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': input_string}]

        response = client.chat.completions.create(model='deepseek-reasoner', messages=messages,
                                                  stream=False, temperature=0.000001, max_tokens=8000)

        output_string = response.choices[0].message.content

        return output_string

    # 内部函数定义好之后，开始对于原始的脑测量信息，进行遍历
    brain_information_row_data = pd.read_excel('dataset/information_of_brain_row_data.xlsx')
    for i in tqdm(range(brain_information_row_data.shape[0]), desc='开始运行deepseek的reasoner模型，解析脑测量变量'):

        input_string = brain_information_row_data.loc[i, '脑测量的含义']

        for re_try in range(5):
            # 给大模型五次输出的机会，try五次
            brain_structure = 'nan'
            chinese_meaning = 'nan'
            try:
                output_string = connect_deepseek(input_string)
                brain_structure, chinese_meaning = output_string.split('-----')
                # 如果大模型的输出正确，那么可以打破内层循环，然后可以保存结果
                break
            except Exception as e:
                print('大模型输出异常：', e)

        # 开始赋值
        brain_information_row_data.loc[i, '脑测量的名词'] = brain_structure
        brain_information_row_data.loc[i, '脑测量的中文含义'] = chinese_meaning

        # 每运行10个就保存一次
        if i % 10 == 0:
            brain_information_row_data.to_excel('dataset/brain_information_result_data.xlsx', index=False)

    # 所有的运行完后，再保存
    brain_information_row_data.to_excel('dataset/brain_information_result_data.xlsx', index=False)
