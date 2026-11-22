"""Offline formal-coach geometry/screenshots with fake cached connections, no network."""
import argparse
import os
import json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from photocoach import api_coach
from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
from photocoach.app import MainWindow, configure_appearance
from photocoach.settings import ApiSettings, credential_target
from photocoach.connections import ConnectionStore, ConnectionBook, connection_from_settings
from photocoach.model_catalog import ModelCatalogStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    app=QApplication([])
    assert app.platformName()=='offscreen', 'This is simulated-layout verification only'
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');configure_appearance(app)
    store=ConnectionStore(args.output/'fixture-settings.json')
    book=ConnectionBook(budget_settings={'call_limit':5})
    for cid,config in [('gemini',ApiConfig(GEMINI_ENDPOINT,'gemini-3.5-flash-lite','gemini')),
                       ('openai',ApiConfig(model='gpt-6-luna'))]:
        book=book.with_connection(connection_from_settings(ApiSettings(config,{},key_source='session'),connection_id=cid))
    book=book.with_active('gemini');store.save(book)
    cache=ModelCatalogStore(args.output/'model_catalog.json')
    for c in book.connections:
        cache.update_provider_models(c.provider,[c.config.model,c.config.model+'-future','o3-mini'] if c.provider=='openai' else [c.config.model,c.config.model+'-future'],credential_target(c.config))
    def forbidden(*a,**k):raise AssertionError('No network or credentials allowed')
    api_coach._https_transport=forbidden
    class Vault:
        read=forbidden
    w=MainWindow(settings_store=store,credentials=Vault(),dev_mode=False)
    w.network_pool.start=forbidden
    w.coach_collapse.setChecked(False)
    w.goal.setText('Preserve natural light')
    w.question.setText('How should I lift the hillside?')
    w.show();w._show_coach_selection();report=[]
    for size in ((1280,800),(1920,1080)):
        w.resize(*size)
        for cid in ('gemini','openai'):
            w._choose_coach_provider(w.coach_provider_picker.findData(cid))
            for _ in range(5):app.processEvents()
            heights=w.coach_panel.height()
            dialog = w.coach_selection
            for widget in (w.coach_provider_picker,w.coach_model_picker,w.coach_favorite,w.coach_connection_button):
                assert widget.isVisible()
                assert dialog.rect().contains(widget.mapTo(dialog,widget.rect().topLeft()))
                assert dialog.rect().contains(widget.mapTo(dialog,widget.rect().bottomRight()))
                assert widget.height()>=24
            w.coach_favorite.click();app.processEvents()
            assert w.coach_panel.height()==heights
            assert w.goal.text()=='Preserve natural light' and w.question.text()=='How should I lift the hillside?'
            i=w.coach_model_picker.findData('o3-mini');assert i==-1
            name=f'{size[0]}x{size[1]}-{cid}.png';assert w.grab().save(str(args.output/name))
            w.coach_model_picker.showPopup();app.processEvents()
            popup=f'{size[0]}x{size[1]}-{cid}-models.png'
            assert w.coach_model_picker.view().window().grab().save(str(args.output/popup))
            w.coach_model_picker.hidePopup();app.processEvents()
            report.append(dict(window=size,dpr=w.devicePixelRatioF(),provider=cid,screenshot=name,popup=popup))
    assert w.api_budget.attempts==w.coach.calls==0
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    w.coach_selection.close()
    w._close_approved=True;w.close()
    print(json.dumps(report))


if __name__=='__main__':main()
