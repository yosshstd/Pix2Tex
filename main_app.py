import const

import time
import streamlit as st
st.set_page_config(**const.SET_PAGE_CONFIG)
st.markdown(const.HIDE_ST_STYLE, unsafe_allow_html=True)
from PIL import Image
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
from streamlit_paste_button import paste_image_button as pbutton
import latex2mathml.converter



def main():
    ''''''
    predicted_formula = ''
    st.markdown(f'<h1 style="text-align:center;">Pix2Tex App</h1>', unsafe_allow_html=True)

    # Load the model cached
    @st.cache_resource
    def load_model():
        processor = TrOCRProcessor.from_pretrained('breezedeus/pix2text-mfr')
        model = ORTModelForVision2Seq.from_pretrained('breezedeus/pix2text-mfr', use_cache=False)
        return processor, model
    processor, model = load_model()
    ''''''
    
    col1, col2 = st.columns([1, 2])
    img_source = col1.radio('Image Source', ('Paste', 'Upload'), help='You can paste an mathematical formula image from clipboard or upload an image from your local machine.')
    if img_source == 'Paste':
        with col1:
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
    else:
        with st.spinner('Loading...'):
            start_time = time.time()
            default_image = Image.open(const.DEFAULT_IMAGE).convert('RGB')
            col2.image(default_image, caption='Default image', use_column_width=True)
            inputs = processor(default_image, return_tensors='pt')
            outputs = model.generate(**inputs)
            predicted_formula = processor.decode(outputs[0], skip_special_tokens=True)
            st.success(f'Elapsed time: {time.time()-start_time:.2f} [sec]')


    col1, col2 = st.columns(2)
    col1.text_area('LaTeX Editer', value='$$\n'+predicted_formula+'\n$$', height=200, key='input_latex')
    col2.markdown('<small>Preview</small>', unsafe_allow_html=True, help='You can preview the mathematical formula here \n if the model can recognize the formula correctly.')
    col2.latex(st.session_state.input_latex.replace('$', ''))

    col1, col2 = st.columns(2)
    col1.expander('Copy (LaTeX)').code(st.session_state.input_latex, language='latex')
    col2.expander('Copy (MathML)').code(latex2mathml.converter.convert(st.session_state.input_latex.replace('$', '')))

    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;">Pix2Tex App</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;font-size:12px;opacity:0.7;">This is a demo app of <a href="https://huggingface.co/breezedeus/pix2text-mfr" target="_blank">Mathematical Formula Recognition</a><br>'
        '(©️ 2022 <a href="https://www.breezedeus.com/join-group" target="_blank">BreezeDeus</a>｜<a href="https://github.com/breezedeus/Pix2Text/blob/main/LICENSE" target="_blank">MIT License</a>).<br></div>',
        unsafe_allow_html=True
    )
    st.markdown('<br>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()



# To-Do
    # - [ ] Add on-change event for the Latex Preview
    # - [ ] change copy botton to a better one