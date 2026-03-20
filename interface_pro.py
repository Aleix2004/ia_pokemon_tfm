import flet as ft
import requests
import asyncio

API_URL = "http://localhost:8000/predict"

class PokemonBattleApp(ft.Column):
    def __init__(self):
        super().__init__()
        self.ia_hp, self.rival_hp = 100, 100
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER # Centrado horizontal

        # Escenario
        self.bg = ft.Image(src="https://play.pokemonshowdown.com/fx/bg-forest.png", width=800, height=450, fit=ft.ImageFit.COVER)
        
        # Sprites
        self.ia_sprite = ft.Image(src="https://play.pokemonshowdown.com/sprites/ani-back/raichu.gif", width=250, bottom=30, left=80, animate_offset=ft.Animation(400, "ease_out_back"))
        self.rival_sprite = ft.Image(src="https://play.pokemonshowdown.com/sprites/ani/dragonite.gif", width=180, top=60, right=100, animate_opacity=ft.Animation(300, "bounce_in"))

        # UI
        self.rival_bar = ft.ProgressBar(value=1.0, width=200, color="green")
        self.ia_bar = ft.ProgressBar(value=1.0, width=200, color="green")
        self.status_text = ft.Text("¡SISTEMA IA LISTO!", color="black", size=22, weight="bold")

    def build(self):
        self.controls = [
            ft.Stack([
                self.bg,
                ft.Container(content=ft.Column([ft.Text("DRAGONITE", weight="bold", color="black"), self.rival_bar]), top=40, right=40, bgcolor="#CCFFFFFF", padding=10, border_radius=10),
                ft.Container(content=ft.Column([ft.Text("RAICHU (IA)", weight="bold", color="black"), self.ia_bar]), bottom=150, left=40, bgcolor="#CCFFFFFF", padding=10, border_radius=10),
                self.rival_sprite, self.ia_sprite,
            ], width=800, height=450),
            ft.Container(
                content=ft.Row([
                    ft.ElevatedButton("🧠 ACCIÓN IA", on_click=self.ia_turn, bgcolor="red", color="white", scale=1.3, height=50),
                    ft.VerticalDivider(width=20),
                    self.status_text
                ], alignment=ft.MainAxisAlignment.CENTER),
                padding=30
            )
        ]
        return self

    async def ia_turn(self, e):
        try:
            res = requests.post(API_URL, json={"ia_hp": int(self.ia_hp), "rival_hp": int(self.rival_hp)}, timeout=3)
            move = res.json().get("move_name", "Ataque").upper()
            self.status_text.value = f"IA USÓ: {move}"
            self.update()

            # Animaciones
            self.ia_sprite.offset = ft.Offset(0.4, -0.4)
            self.ia_sprite.update()
            await asyncio.sleep(0.3)
            self.ia_sprite.offset = ft.Offset(0, 0)
            self.ia_sprite.update()

            for _ in range(3):
                self.rival_sprite.opacity = 0; self.rival_sprite.update()
                await asyncio.sleep(0.1)
                self.rival_sprite.opacity = 1; self.rival_sprite.update()
                await asyncio.sleep(0.1)

            self.rival_hp -= 15
            self.rival_bar.value = max(0, self.rival_hp / 100)
            if self.rival_hp < 35: self.rival_bar.color = "red"
            self.update()
        except:
            self.status_text.value = "ERROR: ¿API ONLINE?"
            self.update()

def main(page: ft.Page):
    page.title = "IA Pokémon TFM - Pro Engine"
    page.window_width, page.window_height = 850, 650 # Ventana más ajustada
    page.window_resizable = False # Que no se deforme
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.add(PokemonBattleApp().build())

ft.app(target=main)