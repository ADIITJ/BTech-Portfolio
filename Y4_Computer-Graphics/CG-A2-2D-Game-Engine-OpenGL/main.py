from OpenGL.GL import *
from utils.window_manager import Window
from game import Game
import imgui
from imgui.integrations.glfw import GlfwRenderer

class App:
    def __init__(self, width, height):
        self.window = Window(width, height)
        glClearColor(0, 0, 0, 1)
        self.game = Game(height, width)
        
        # Initialize imgui
        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.window.window)

    def RenderLoop(self):
        while self.window.IsOpen():
            inputs, time = self.window.StartFrame(0.2, 0.2, 0.2, 1.0)
            self.game.ProcessFrame(inputs, time)
            glDisable(GL_DEPTH_TEST)
            
            # Render imgui
            imgui.new_frame()
            
            # Show Game Over text if game is over
            if self.game.game_over:
                imgui.set_next_window_position(300, 400)
                imgui.set_next_window_size(400, 200)
                imgui.begin("Game Over", flags=imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.0, 0.0, 1.0)  # Red text
                imgui.set_window_font_scale(3.0)
                imgui.text("GAME OVER!")
                imgui.set_window_font_scale(1.5)
                imgui.pop_style_color(1)
                imgui.text(f"\nTotal Time: {self.game.elapsed_time:.1f}s")
                imgui.text(f"Keys Collected: {sum(self.game.collect_key_check)}/7")
                imgui.text(f"Lives Remaining: {self.game.player_lives}")
                imgui.text("\nPress ESC to exit")
                imgui.end()
            
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())
            
            self.window.EndFrame()
        
        self.imgui_impl.shutdown()
        self.window.Close()

if __name__ == "__main__":
    app = App(1000, 1000)
    app.RenderLoop()
