# from __future__ import print_function
import ipywidgets
import bokeh
from . import utils
from . import config

extend = utils.extend

def kwargser(d):
    args = dict(d)
    if 'kwargs' in args:
        kwargs = args['kwargs']
        del args['kwargs']
        args.update(kwargs)
    return args

def toggle(description, value, **kwargs):  # pylint: disable=unused-argument
    return ipywidgets.ToggleButton(**kwargser(locals()))

def toggles(description, options, value, **kwopts):  # pylint: disable=unused-argument
    return ipywidgets.ToggleButtons(**kwargser(locals()))

def dropdown(description, options, value, **kwargs):  # pylint: disable=unused-argument
    return ipywidgets.Dropdown(**kwargser(locals()))

def selectmultiple(description, options, value, **kwargs):  # pylint: disable=unused-argument
    return ipywidgets.SelectMultiple(**kwargser(locals()))

def slider(description, min, max, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return ipywidgets.IntSlider(**kwargser(locals()))

def rangeslider(description, min, max, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return ipywidgets.IntRangeSlider(**kwargser(locals()))

def sliderf(description, min, max, step, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return ipywidgets.FloatSlider(**kwargser(locals()))

def progress(min, max, step, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return ipywidgets.IntProgress(**kwargser(locals()))

def itext(min, max, value, **kwargs):  # pylint: disable=unused-argument, redefined-builtin
    return ipywidgets.BoundedIntText(**kwargser(locals()))

def wrap_id_text(dom_id, value=''):
    value = "<span class='{}'>{}</span>".format(dom_id, value) if dom_id is not None else value
    return value

def text(dom_id=None, value=''):
    return ipywidgets.HTML(value=wrap_id_text(dom_id, value), placeholder='', description='')

def button(description):
    return ipywidgets.Button(**kwargser(locals()))

def glyph_hover_js_code(element_id, id_name, text_name, glyph_name='glyph', glyph_data='glyph_data'):
    return """
        var indices = cb_data.index['1d'].indices;
        var current_id = -1;
        if (indices.length > 0) {
            var index = indices[0];
            var id = parseInt(""" + glyph_name + """.data.""" + id_name + """[index]);
            if (id !== current_id) {
                current_id = id;
                var text = """ + glyph_data + """.data.""" + text_name + """[id];
                document.getElementsByClassName('""" + element_id + """')[0].innerText = 'ID ' + id.toString() + ': ' + text;
            }
    }
    """
def glyph_hover_callback2(glyph_source, glyph_id, text_ids, text, element_id):
    source = bokeh.models.ColumnDataSource(dict(text_id=text_ids, text=text))
    code = glyph_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
    callback = bokeh.models.CustomJS(args={'glyph': glyph_source, 'glyph_data': source}, code=code)
    return callback

def glyph_hover_callback(glyph_source, glyph_id, text_source, element_id):
    code = glyph_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
    callback = bokeh.models.CustomJS(args={'glyph': glyph_source, 'glyph_data': text_source}, code=code)
    return callback

def aggregate_function_widget(**kwopts):
    default_opts = dict(
        options=['mean', 'sum', 'std', 'min', 'max'],
        value='mean',
        description='Aggregate',
        layout=ipywidgets.Layout(width='200px')
    )
    return ipywidgets.Dropdown(**extend(default_opts, kwopts))

def years_widget(**kwopts):
    default_opts = dict(
        options=[],
        value=None,
        description='Year',
        layout=ipywidgets.Layout(width='200px')
    )
    return ipywidgets.Dropdown(**extend(default_opts, kwopts))


def plot_style_widget(**kwopts):
    default_opts = dict(
        options=[ x for x in config.MATPLOTLIB_PLOT_STYLES if 'seaborn' in x ],
        value='seaborn-pastel',
        description='Style:',
        layout=ipywidgets.Layout(width='200px')
    )
    return ipywidgets.Dropdown(**extend(default_opts, kwopts))

def increment_button(target_control, max_value, label='>>', increment=1):

    def f(_):
        target_control.value = (target_control.value + increment) % max_value

    return ipywidgets.Button(description=label, callback=f)

def _get_field_values(documents, field, as_tuple=False, query=None):
    items = documents.query(query) if query is not None else documents
    unique_values = sorted(list(items[field].unique()))
    if as_tuple:
        unique_values = [ (str(x).title(), x) for x in unique_values ]
    return unique_values

def generate_field_filters(documents, opts):
    filters = []
    for opt in opts:  # if opt['type'] == 'multiselect':
        options =  opt.get('options', _get_field_values(documents, opt['field'], as_tuple=True, query=opt.get('query', None)))
        description = opt.get('description', '')
        rows = min(4, len(options))
        gf = extend(opt, widget=selectmultiple(description, options, value=(), rows=rows))
        filters.append(gf)
    return filters
