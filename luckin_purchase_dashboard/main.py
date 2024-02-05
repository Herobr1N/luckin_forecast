import dash
import dash_auth
from dash import Dash, html, dcc, Input, Output, callback
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from utils import logger, today
import data_syc as ds

logger.info("Init")
ds.load()
app = Dash(__name__, use_pages=True)


auth = dash_auth.BasicAuth(app, ds.auth_info)


def layout_main():
    return html.Div([
        html.H1(id='title'),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000
        ),
        dash.page_container

    ])


logger.info("Init Finish")
app.layout = layout_main


@callback(Output('title', 'children'),
          Input('interval-component', 'n_intervals'),
          prevent_initial_call=True)
def update(n_intervals):
    dtime = datetime.now()
    if (dtime.hour == 9) & (dtime.minute == 0) & (dtime.second == 0):
        logger.info("Reload Data")
        ds.load(dtime.date())
    return f"Hello, {dtime.strftime('%Y-%m-%d %H:%M:%S')}"


if __name__ == '__main__':
    # 每日九点自动更新后台数据
    sched = BackgroundScheduler(timezone='Asia/Shanghai')

    @sched.scheduled_job('cron', day_of_week='mon-sun', hour=9, minute=00)
    def scheduled_job():
        print("Updating data for: {}".format(datetime.today().date().strftime("%m-%d-%Y")))
        ds.load(datetime.now().date())

    sched.start()

    app.run_server(debug=True, host='0.0.0.0', port='7777', use_reloader=False)

