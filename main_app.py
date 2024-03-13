import const

import time
import streamlit as st
st.set_page_config(**const.SET_PAGE_CONFIG)
st.markdown(const.HIDE_ST_STYLE, unsafe_allow_html=True)
from PIL import Image
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
from st_copy_to_clipboard import st_copy_to_clipboard
from streamlit_paste_button import paste_image_button as pbutton



def main():
    ''''''
    predicted_formula = ''
    st.markdown(f"<h1 style='text-align:center;'>Pix2Tex App</h1>", unsafe_allow_html=True)
    st.sidebar.caption(
        'This is a demo app of [Mathematical Formula Recognition](https://huggingface.co/breezedeus/pix2text-mfr).\n \
        The model is based on the [TrOCR](https://huggingface.co/models?search=trOCR) architecture and was retraiend on a dataset of mathematical formula images.'
    )
    # Load the model cached
    @st.cache_resource
    def load_model():
        processor = TrOCRProcessor.from_pretrained('breezedeus/pix2text-mfr')
        model = ORTModelForVision2Seq.from_pretrained('breezedeus/pix2text-mfr', use_cache=False)
        return processor, model
    processor, model = load_model()
    ''''''
    
    col1, col2 = st.columns([1, 2])
    img_source = col1.radio('Image Source', ('Paste', 'Upload'))
    if img_source == 'Paste':
        out = pbutton('Paste an image').image_data
        try:
            image_data = out.convert('RGB')
        except:
            image_data = None
    elif img_source == 'Upload':
        image_file = col1.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        try :
            image_data = Image.open(image_file).convert('RGB')
        except:
            image_data = None
    


    if image_data is not None:

        # Perform OCR
        with st.spinner('Loading...'):
            start_time = time.time()
            col2.image(image_data, caption='Uploaded image', use_column_width=True)
            inputs = processor(image_data, return_tensors='pt')
            outputs = model.generate(**inputs)
            predicted_formula = processor.decode(outputs[0], skip_special_tokens=True)
            st.success(f'Elapsed time: {time.time()-start_time:.2f} [sec]')


    # def on_latex_change():
    #     if st.session_state.input_latex:
    #         col2.latex(st.session_state.input_latex.replace('$', ''))
    #     else:
    #         col2.empty()
    col1, col2 = st.columns(2)
    col1.text_area('Latex Editer', value='$$\n'+predicted_formula+'\n$$', height=300, key='input_latex')
    st_copy_to_clipboard(st.session_state.input_latex)
    col2.markdown('<small>Preview</small>', unsafe_allow_html=True)
    col2.latex(st.session_state.input_latex.replace('$', ''))

    st.expander('To-Do').markdown('''
    - [ ] Add on-change event for the Latex Preview
    - [ ] Add copy mathml to clipboard
    ''')
    


if __name__ == '__main__':
    main()

## To-Do:
# - [ ] Add copy to cli