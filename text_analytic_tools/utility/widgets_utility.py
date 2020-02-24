# from __future__ import print_function
import ipywidgets as widgets
from text_analytic_tools.utility.widgets import glyph_hover_js_code, years_widget
from bokeh.models import ColumnDataSource, CustomJS

BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')

class WidgetUtility():

    @staticmethod
    def create_js_callback(axis, attribute, source):
        return CustomJS(args=dict(source=source), code="""
            var data = source.data;
            var start = cb_obj.start;
            var end = cb_obj.end;
            data['""" + axis + """'] = [start + (end - start) / 2];
            data['""" + attribute + """'] = [end - start];
            source.change.emit();
        """)

    @staticmethod
    def glyph_hover_callback(glyph_source, glyph_id, text_ids, text, element_id):
        source = ColumnDataSource(dict(text_id=text_ids, text=text))
        code = glyph_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
        callback = CustomJS(args={'glyph': glyph_source, 'glyph_data': source}, code=code)
        return callback

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # self.__dict__.update(kwargs)

    def create_button(self, description, style=None, callback=None):
        style = style or dict(description_width='initial', button_color='lightgreen')
        button = widgets.Button(description=description, style=style)
        if callback is not None:
            button.on_click(callback)
        return button

    def create_text_widget(self, element_id=None, default_value=''):
        value = "<span class='{}'>{}</span>".format(element_id, default_value) if element_id is not None else ''
        return widgets.HTML(value=value, placeholder='', description='')

    def create_prev_button(self, callback):
        return self.create_button(description="<<", callback=callback)

    def create_next_button(self, callback):
        return self.create_button(description=">>", callback=callback)

    def create_next_id_button(self, name, count):
        that = self

        def f(_):
            control = getattr(that, name, None)
            if control is not None:
                control.value = (control.value + 1) % count

        return self.create_button(description=">>", callback=f)

    def create_prev_id_button(self, name, count):
        that = self

        def f(_):
            control = getattr(that, name, None)
            if control is not None:
                control.value = (control.value - 1) % count

        return self.create_button(description="<<", callback=f)

    def next_topic_id_clicked(self, _):
        self.topic_id.value = (self.topic_id.value + 1) % self.n_topics

    def prev_topic_id_clicked(self, _):
        self.topic_id.value = (self.topic_id.value - 1) % self.n_topics


class TopicWidgets(WidgetUtility):

    def __init__(self, n_topics, years=None, word_count=None, text_id=None):

        self.n_topics = n_topics
        self.text_id = text_id
        self.text = self.create_text_widget(text_id)
        self.year = years_widget(options=years) if years is not None else None
        self.topic_id = self.topic_id_slider(n_topics)

        self.word_count = self.word_count_slider(1, 500) if word_count is not None else None

        self.prev_topic_id = self.create_prev_button(self.prev_topic_id_clicked)
        self.next_topic_id = self.create_next_button(self.next_topic_id_clicked)

class TopTopicWidgets(WidgetUtility):

    def __init__(self, n_topics=0, years=None, aggregates=None, text_id='text_id', layout_algorithms=None):

        self.n_topics = n_topics
        self.text_id = text_id
        self.text = self.create_text_widget(text_id) if text_id is not None else None
        self.year = years_widget(options=years) if years is not None else None

        self.topics_count = self.topic_count_slider(n_topics) if n_topics > 0 else None

        self.aggregate = self.select_aggregate_fn_widget(aggregates, default='mean') if aggregates is not None else None
        self.layout_algorithm = self.layout_algorithm_widget(layout_algorithms, default='Fruchterman-Reingold') \
            if layout_algorithms is not None else None

wf = WidgetUtility()
