from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

class ActionQueryBiaya(Action):
    def name(self) -> Text:
        return "action_query_biaya"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        jenjang = tracker.get_slot("jenjang")
        prodi = tracker.get_slot("prodi")
        
        # Debug: print slots
        print(f"🔍 DEBUG - jenjang slot: '{jenjang}', prodi slot: '{prodi}'")
        
        try:
            url = "http://localhost:3000/biaya"
            params = {}
            
            if jenjang:
                params['jenjang'] = jenjang
            if prodi:
                params['prodi'] = prodi
            
            print(f"🌐 API Call: {url} with params: {params}")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                biaya_list = response.json()
                print(f"📊 API Response: {len(biaya_list)} items found")
                
                if biaya_list:
                    message = "💰 **Informasi Biaya Pascasarjana ITS:**\n\n"
                    
                    for biaya in biaya_list:
                        message += f"🎓 **{biaya['jenjang']} - {biaya['program']}**\n"
                        message += f"   💵 {biaya['biaya_label']}\n"
                        if biaya.get('spi'):
                            message += f"   📊 SPI: {biaya['spi']}\n"
                        if biaya.get('ipits'):
                            message += f"   🏛️ IPITS: {biaya['ipits']}\n"
                        message += "\n"
                else:
                    message = "Maaf, data biaya tidak ditemukan."
            else:
                message = f"Maaf, terjadi kendala dalam mengambil data biaya. Status: {response.status_code}"
                
        except Exception as e:
            print(f"❌ Error: {e}")
            message = "Maaf, terjadi kesalahan sistem."
        
        dispatcher.utter_message(text=message)
        return []

class ActionListProdiByFakultas(Action):
    def name(self) -> Text:
        return "action_list_prodi_by_fakultas"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        fakultas = tracker.get_slot("fakultas")
        
        # Debug: print slot
        print(f"🔍 DEBUG - fakultas slot: '{fakultas}'")
        
        if not fakultas:
            dispatcher.utter_message(response="utter_ask_fakultas")
            return []
        
        try:
            url = "http://localhost:3000/prodi"
            params = {"fakultas": fakultas}
            
            print(f"🌐 API Call: {url} with params: {params}")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                prodi_list = response.json()
                print(f"📊 API Response: {len(prodi_list)} items found")
                
                if prodi_list:
                    message = f"🏛️ **Program Studi di Fakultas {fakultas}:**\n\n"
                    for prodi in prodi_list:
                        message += f"   • {prodi['prodi']}\n"
                else:
                    message = f"Maaf, tidak ada program studi ditemukan untuk fakultas '{fakultas}'."
            else:
                message = f"Maaf, terjadi kendala dalam mengambil data program studi. Status: {response.status_code}"
                
        except Exception as e:
            print(f"❌ Error: {e}")
            message = "Maaf, terjadi kesalahan sistem."
        
        dispatcher.utter_message(text=message)
        return []

class ActionLookupFakultasByProdi(Action):
    def name(self) -> Text:
        return "action_lookup_fakultas_by_prodi"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        prodi = tracker.get_slot("prodi")
        
        # Debug: print slot
        print(f"🔍 DEBUG - prodi slot: '{prodi}'")
        
        if not prodi:
            dispatcher.utter_message(response="utter_ask_prodi")
            return []
        
        try:
            url = "http://localhost:3000/fakultas"
            params = {"prodi": prodi}
            
            print(f"🌐 API Call: {url} with params: {params}")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                print(f"📊 API Response: {result}")
                
                if result and 'fakultas' in result:
                    fakultas = result['fakultas']
                    message = f"🎓 **Program Studi: {prodi}**\n\n🏛️ **Fakultas:** {fakultas}"
                else:
                    message = f"Maaf, program studi '{prodi}' tidak ditemukan."
            else:
                message = f"Maaf, terjadi kendala dalam mencari data program studi. Status: {response.status_code}"
                
        except Exception as e:
            print(f"❌ Error: {e}")
            message = "Maaf, terjadi kesalahan sistem."
        
        dispatcher.utter_message(text=message)
        return []