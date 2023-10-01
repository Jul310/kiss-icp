import matplotlib

_TEX_COLUMN_WIDTH = 422.52348
_TEX_FONTSIZE = "11"

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = _TEX_FONTSIZE
 
def _get_figsize(columnwidth, wf=0.5, hf=(5.**0.5-1.0)/2.0, ):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth * wf 
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * hf      # height in inches
    return [fig_width, fig_height]


def get_figsize(wf=None, hf=None):
    if not wf or not hf:
        return (_get_figsize(_TEX_COLUMN_WIDTH))
    return _get_figsize(_TEX_COLUMN_WIDTH, wf, hf)