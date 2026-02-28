import numpy as np
import ast

from utils.graphics import Object, Camera, Shader
from assets.shaders.shaders import object_shader
from assets.objects.objects import (
    playerProps, enemyProps, entryProps, exitProps,
    backgroundProps2, sunProps, cloudprops,
    keyProps, doorprops
)


class Game:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        # Game state
        self.state = "GAME"
        self.screen = 0
        self._level_initialized = False
        self.game_over = False

        # Door state machine
        self.entry_door_state = "closed"  # closed, opening, open, closing
        self.exit_door_state = "closed"
        self.entry_door_timer = 0.0
        self.exit_door_timer = 0.0
        self.entry_door_initial_pos = None
        self.exit_door_initial_pos = None
        self.door_open_distance = 60.0  # How far door moves to left
        self.door_animation_time = 2.0  # Time for door to open/close
        self.chuck_visible = False  # Chuck starts invisible at game start

        # Rendering
        self.camera = Camera(height, width)
        self.shaders = [Shader(object_shader["vertex_shader"], object_shader["fragment_shader"])]

        # Gameplay vars
        self.player_health = 100
        self.player_lives = 3
        self.collect_key_check = [False] * 7

        # time tracking
        self.elapsed_time = 0.0
        self._hud_print_accum = 0.0

        # Input debouncing for save/load
        self._save_key_pressed = False
        self._load_key_pressed = False

        # physics flags
        self.is_on_platform = False

        # Physics tuning (you can tune these)
        self.GRAVITY = 120.0           # reduced gravity for better control
        self.JUMP_V = 200.0            # reasonable jump height
        self.MOVE_V = 140.0

        # Collision sizing heuristic
        self.BASE_UNIT = 10.0          # tune if needed (10 is your original implied scale)

    # -----------------------------
    # Level init (only when needed)
    # -----------------------------
    def InitScreen(self, reset_keys=False):
        if reset_keys:
            self.collect_key_check = [False] * 7

        # Background & static props
        self.bg = Object(self.shaders[0], backgroundProps2)
        self.sun = Object(self.shaders[0], sunProps)

        # Entry / Exit
        self.entry = Object(self.shaders[0], entryProps)
        self.entry.properties["position"] = np.array([-460, 320, 0], dtype=np.float32)

        self.exit = Object(self.shaders[0], exitProps)
        self.exit.properties["position"] = np.array([460, -280, 0], dtype=np.float32)

        # Doors
        self.entrydoor = Object(self.shaders[0], doorprops)
        # Start entry door at the entry point
        self.entrydoor.properties["position"] = self.entry.properties["position"] + np.array([0, 0, 20], dtype=np.float32)
        self.entrydoor.properties["velocity"] = np.array([0, 0, 0], dtype=np.float32)
        self.entry_door_initial_pos = self.entrydoor.properties["position"].copy()

        self.exitdoor = Object(self.shaders[0], doorprops)
        # Start exit door at the exit point
        self.exitdoor.properties["position"] = self.exit.properties["position"] + np.array([0, 0, 20], dtype=np.float32)
        self.exitdoor.properties["velocity"] = np.array([0, 0, 0], dtype=np.float32)
        self.exit_door_initial_pos = self.exitdoor.properties["position"].copy()

        # Player
        self.player = Object(self.shaders[0], playerProps)
        # Start Chuck behind the door (negative z = behind)
        self.player.properties["position"] = self.entry.properties["position"] + np.array([0.0, 0.0, -50], dtype=np.float32)
        self.player.properties["velocity"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Start door opening sequence
        self.entry_door_state = "opening"
        self.entry_door_timer = 0.0

        # Clouds + Keys
        self.cloud = []
        self.keys = []
        
        # Add starting platform (stationary) near entry
        start_platform = Object(self.shaders[0], cloudprops)
        start_platform.properties["position"] = np.array([-460, 250, 0], dtype=np.float32)
        start_platform.properties["scale"] = np.array([1.8, 1.5, 1], dtype=np.float32)  # Smaller platform
        start_platform.properties["velocity"] = np.array([0, 0, 0], dtype=np.float32)  # Stationary
        self.cloud.append(start_platform)
        
        # Add invisible key for starting platform (at index 0)
        start_key = Object(self.shaders[0], keyProps)
        start_key.properties["position"] = np.array([0, 0, -1000], dtype=np.float32)
        start_key.properties["scale"] = np.array([0, 0, 0], dtype=np.float32)
        self.keys.append(start_key)
        
        for i in range(7):
            cl = Object(self.shaders[0], cloudprops)
            cl.properties["position"] = np.array(
                [
                    (-self.width / 2) + (i + 1) * (self.width / 8),
                    (self.height / 2) - (self.height / 8) * (i + 1),
                    0,
                ],
                dtype=np.float32,
            )
            cl.properties["scale"] = np.array([2, 2, 1], dtype=np.float32)
            cl.properties["velocity"] = np.array([0, 125 + 4 * i, 0], dtype=np.float32)
            self.cloud.append(cl)

            key = Object(self.shaders[0], keyProps)
            key.properties["position"] = cl.properties["position"] + np.array([0, 15, 15], dtype=np.float32)
            key.properties["scale"] = np.array([3, 3, 1], dtype=np.float32)

            if self.collect_key_check[i]:
                key.properties["scale"] = np.array([0, 0, 0], dtype=np.float32)

            self.keys.append(key)

        # Enemies
        self.enemies = []
        for _ in range(5):
            ene = Object(self.shaders[0], enemyProps)
            ene.properties["position"] = np.array(
                [
                    np.random.uniform(-(self.width / 2) + 200, (self.width / 2) - 200),
                    np.random.uniform(-self.height / 2 + 100, self.height / 2 - 100),
                    50,
                ],
                dtype=np.float32,
            )
            ene.properties["velocity"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.enemies.append(ene)

        self._level_initialized = True

    # -----------------------------
    # Save / Load (no imgui)
    # -----------------------------
    def save_game(self):
        state_data = {
            "screen": int(self.screen),
            "player_lives": int(self.player_lives),
            "player_health": int(self.player_health),
            "collect_key_check": list(self.collect_key_check),
            "elapsed_time": float(self.elapsed_time),
        }
        with open("savegame.txt", "w") as f:
            f.write(repr(state_data))
        print("Saved game -> savegame.txt")

    def load_game(self):
        try:
            with open("savegame.txt", "r") as f:
                data = ast.literal_eval(f.read())

            self.screen = int(data.get("screen", 0))
            self.player_lives = int(data.get("player_lives", 3))
            self.player_health = int(data.get("player_health", 100))
            self.collect_key_check = list(data.get("collect_key_check", [False] * 7))
            self.elapsed_time = float(data.get("elapsed_time", 0.0))

            self.state = "GAME"
            self._level_initialized = False
            print("Loaded game from savegame.txt")

        except Exception as e:
            print("Load game failed:", e)

    # -----------------------------
    # Core loop
    # -----------------------------
    def ProcessFrame(self, inputs, time_dict):
        dt = time_dict["deltaTime"]

        # If game over, just show empty screen
        if self.game_over:
            return

        # Window provides: "1","2","W","A","S","D","SPACE","UP","DOWN","LEFT","RIGHT"
        # Save/Load with debouncing
        if "1" in inputs and not self._save_key_pressed:
            self.save_game()
            self._save_key_pressed = True
        elif "1" not in inputs:
            self._save_key_pressed = False

        if "2" in inputs and not self._load_key_pressed:
            self.load_game()
            self._load_key_pressed = True
        elif "2" not in inputs:
            self._load_key_pressed = False

        if not self._level_initialized:
            self.InitScreen(reset_keys=False)

        # Failsafe
        if not hasattr(self, "player") or not hasattr(self, "bg"):
            self._level_initialized = False
            self.InitScreen(reset_keys=False)

        self.elapsed_time += dt

        self.UpdateScene(inputs, time_dict)
        self.DrawScene()
        self._print_hud_to_console(dt)

    def _print_hud_to_console(self, dt):
        self._hud_print_accum += dt
        if self._hud_print_accum >= 1.0:
            self._hud_print_accum = 0.0
            print(
                f"Lives={self.player_lives} | Health={self.player_health} | "
                f"Level={self.screen+1} | Keys={sum(self.collect_key_check)} | "
                f"Time={self.elapsed_time:.1f}s"
            )

    # -----------------------------
    # Helper: clamp X only (do not clamp Y)
    # -----------------------------
    def _clamp_player_x_only(self):
        pos = self.player.properties["position"]
        pos[0] = np.clip(pos[0], -self.width / 2, self.width / 2)
        self.player.properties["position"] = pos

    # -----------------------------
    # Gameplay update
    # -----------------------------
    def UpdateScene(self, inputs, time_dict):
        dt = time_dict["deltaTime"]

        # ========== DOOR ANIMATION STATE MACHINE ==========
        # Entry door opening sequence (at start of game)
        if self.entry_door_state == "opening":
            self.entry_door_timer += dt
            progress = min(self.entry_door_timer / self.door_animation_time, 1.0)
            # Move door to the left
            self.entrydoor.properties["position"][0] = self.entry_door_initial_pos[0] - (self.door_open_distance * progress)
            
            if progress >= 1.0:
                self.entry_door_state = "open"
                self.entry_door_timer = 0.0
                # Make Chuck visible (bring to front)
                self.player.properties["position"][2] = 50
                self.chuck_visible = True
                
        elif self.entry_door_state == "open":
            # Wait for Chuck to move away
            self.entry_door_timer += dt
            if self.entry_door_timer > 1.0:  # Wait 1 second
                self.entry_door_state = "closing"
                self.entry_door_timer = 0.0
                
        elif self.entry_door_state == "closing":
            self.entry_door_timer += dt
            progress = min(self.entry_door_timer / self.door_animation_time, 1.0)
            # Move door back to initial position
            current_x = self.entry_door_initial_pos[0] - self.door_open_distance
            self.entrydoor.properties["position"][0] = current_x + (self.door_open_distance * progress)
            
            if progress >= 1.0:
                self.entry_door_state = "closed"
                self.entrydoor.properties["position"][0] = self.entry_door_initial_pos[0]

        # Exit door sequence (when Chuck reaches exit)
        if self.exit_door_state == "opening":
            self.exit_door_timer += dt
            progress = min(self.exit_door_timer / self.door_animation_time, 1.0)
            # Move door to the left
            self.exitdoor.properties["position"][0] = self.exit_door_initial_pos[0] - (self.door_open_distance * progress)
            
            if progress >= 0.5 and self.chuck_visible:
                # Make Chuck disappear halfway through door opening
                self.player.properties["position"][2] = -50
                self.chuck_visible = False
            
            if progress >= 1.0:
                self.exit_door_state = "open"
                self.exit_door_timer = 0.0
                
        elif self.exit_door_state == "open":
            # Wait before closing
            self.exit_door_timer += dt
            if self.exit_door_timer > 0.5:
                self.exit_door_state = "closing"
                self.exit_door_timer = 0.0
                
        elif self.exit_door_state == "closing":
            self.exit_door_timer += dt
            progress = min(self.exit_door_timer / self.door_animation_time, 1.0)
            # Move door back to initial position
            current_x = self.exit_door_initial_pos[0] - self.door_open_distance
            self.exitdoor.properties["position"][0] = current_x + (self.door_open_distance * progress)
            
            if progress >= 1.0:
                self.exit_door_state = "closed"
                self.exitdoor.properties["position"][0] = self.exit_door_initial_pos[0]
                # Game Over!
                self.game_over = True
                print("\n" + "="*50)
                print("GAME OVER!")
                print(f"Total Time: {self.elapsed_time:.1f}s")
                print(f"Keys Collected: {sum(self.collect_key_check)}/7")
                print("="*50 + "\n")

        # Only allow movement if Chuck is visible
        if not self.chuck_visible:
            return

        # Horizontal movement (support both WASD and arrow keys)
        self.player.properties["velocity"][0] = 0.0
        if "A" in inputs or "LEFT" in inputs:
            self.player.properties["velocity"][0] = -self.MOVE_V
        if "D" in inputs or "RIGHT" in inputs:
            self.player.properties["velocity"][0] = self.MOVE_V

        # Update clouds (move platforms)
        for i in range(len(self.cloud)):
            cloud = self.cloud[i]

            # Only move platforms that have velocity (skip starting platform)
            if np.linalg.norm(cloud.properties["velocity"]) > 0:
                # Check boundaries and reverse direction if needed
                next_y = cloud.properties["position"][1] + cloud.properties["velocity"][1] * dt
                if next_y > (self.height / 2) - 50 or next_y < (-self.height / 2) + 50:
                    cloud.properties["velocity"][1] *= -1
                
                cloud.properties["position"] += cloud.properties["velocity"] * dt
                
                # Keep platforms strictly within bounds
                cloud.properties["position"][1] = np.clip(
                    cloud.properties["position"][1],
                    (-self.height / 2) + 50,
                    (self.height / 2) - 50
                )

        # -----------------------------
        # Platform collision (FIXED)
        # Use bottom-of-player vs top-of-platform
        # -----------------------------
        self.is_on_platform = False
        player_pos = self.player.properties["position"]
        player_vel = self.player.properties["velocity"]

        # approximate half sizes
        player_half_w = 0.5 * self.BASE_UNIT * float(self.player.properties["scale"][0])
        player_half_h = 0.5 * self.BASE_UNIT * float(self.player.properties["scale"][1])

        player_bottom = float(player_pos[1]) - player_half_h

        landed_on = None
        for cloud in self.cloud:
            cloud_pos = cloud.properties["position"]
            cloud_half_w = 0.5 * self.BASE_UNIT * float(cloud.properties["scale"][0])
            cloud_half_h = 0.5 * self.BASE_UNIT * float(cloud.properties["scale"][1])

            platform_top = float(cloud_pos[1]) + cloud_half_h

            # AABB overlap in X (reduced width for tighter landing - 0.8 factor makes it 80% of platform width)
            if abs(float(player_pos[0]) - float(cloud_pos[0])) <= (player_half_w + cloud_half_w * 0.8):
                # Landing condition (tighter vertical tolerance)
                if player_vel[1] <= 0.0 and (platform_top - 5.0) <= player_bottom <= (platform_top + 8.0):
                    self.is_on_platform = True
                    landed_on = cloud

                    # Snap onto platform - place bird exactly on surface
                    player_pos[1] = platform_top + player_half_h + 1.0
                    player_vel[1] = 0.0
                    break

        # Optional: move with platform (clouds move in Y)
        if self.is_on_platform and landed_on is not None:
            player_pos[1] += float(landed_on.properties["velocity"][1]) * dt

        # ========== CLIFF COLLISION (Task 1) ==========
        # Check if Chuck is on the entry cliff (left cliff)
        entry_half_w = 0.5 * self.BASE_UNIT * float(self.entry.properties["scale"][0])
        entry_half_h = 0.5 * self.BASE_UNIT * float(self.entry.properties["scale"][1])
        entry_top = float(self.entry.properties["position"][1]) + entry_half_h
        
        # Check if Chuck's x-coordinate is within cliff width
        if abs(float(player_pos[0]) - float(self.entry.properties["position"][0])) <= (player_half_w + entry_half_w):
            # Landing on entry cliff
            if player_vel[1] <= 0.0 and (entry_top - 5.0) <= player_bottom <= (entry_top + 8.0):
                self.is_on_platform = True
                player_pos[1] = entry_top + player_half_h + 1.0
                player_vel[1] = 0.0

        # Check if Chuck is on the exit cliff (right cliff)
        exit_half_w = 0.5 * self.BASE_UNIT * float(self.exit.properties["scale"][0])
        exit_half_h = 0.5 * self.BASE_UNIT * float(self.exit.properties["scale"][1])
        exit_top = float(self.exit.properties["position"][1]) + exit_half_h
        
        # Check if Chuck's x-coordinate is within cliff width (same as entry cliff)
        if abs(float(player_pos[0]) - float(self.exit.properties["position"][0])) <= (player_half_w + exit_half_w):
            # Landing on exit cliff
            if player_vel[1] <= 0.0 and (exit_top - 5.0) <= player_bottom <= (exit_top + 8.0):
                self.is_on_platform = True
                player_pos[1] = exit_top + player_half_h + 1.0
                player_vel[1] = 0.0
                
                # Trigger exit door sequence when landing on exit cliff
                if self.exit_door_state == "closed":
                    self.exit_door_state = "opening"
                    self.exit_door_timer = 0.0
                    print("Chuck reached the exit! Door opening...")

        # Gravity
        if not self.is_on_platform:
            player_vel[1] -= self.GRAVITY * dt

        # Jump (support both SPACE and UP arrow)
        if self.is_on_platform and ("SPACE" in inputs or "UP" in inputs):
            player_vel[1] = self.JUMP_V

        # Integrate player
        self.player.properties["position"] += self.player.properties["velocity"] * dt

        # Clamp X only (do NOT clamp Y)
        self._clamp_player_x_only()

        # Update key positions and check collection (after platform detection)
        for i in range(len(self.cloud)):
            cloud = self.cloud[i]
            if i < len(self.keys):
                key = self.keys[i]
                key.properties["position"] = cloud.properties["position"] + np.array([0, 20, 15], dtype=np.float32)

                # Collect key - only for moving platforms (skip index 0 starting platform)
                # and only when on this platform
                if i > 0 and i-1 < len(self.collect_key_check):
                    key_index = i - 1  # Adjust for starting platform offset
                    if not self.collect_key_check[key_index] and self.is_on_platform and landed_on == cloud:
                        self.collect_key_check[key_index] = True
                        key.properties["scale"] = np.array([0, 0, 0], dtype=np.float32)
                        print(f"Key {key_index+1} collected!")

        # Death fall (below screen)
        if self.player.properties["position"][1] < (-self.height / 2) - 200:
            self.player_lives -= 1
            self.player_health = 100
            self.player.properties["position"] = self.entry.properties["position"] + np.array([0.0, 0.0, 50], dtype=np.float32)
            self.player.properties["velocity"][:] = 0.0
            if self.player_lives <= 0:
                print("GAME OVER")
                self.player_lives = 3
                self.player_health = 100
                self.collect_key_check = [False] * 7
                self._level_initialized = False
                return

        # Enemy hit
        for enemy in self.enemies:
            if np.linalg.norm(self.player.properties["position"] - enemy.properties["position"]) < 100:
                self.player_health -= 10
                if self.player_health <= 0:
                    self.player_lives -= 1
                    self.player_health = 100
                    self.player.properties["position"] = self.entry.properties["position"] + np.array([0.0, 0.0, 50], dtype=np.float32)
                    self.player.properties["velocity"][:] = 0.0
                    if self.player_lives <= 0:
                        print("GAME OVER")
                        self.player_lives = 3
                        self.player_health = 100
                        self.collect_key_check = [False] * 7
                        self._level_initialized = False
                        return

    def _next_level(self):
        # Not used in this version - game ends when reaching right cliff
        pass

    # -----------------------------
    # Draw
    # -----------------------------
    def DrawScene(self):
        # If game over, show empty screen (clear screen already done in main loop)
        if self.game_over:
            return
            
        for shader in self.shaders:
            self.camera.Update(shader)

        self.bg.Draw()
        self.sun.Draw()
        self.entry.Draw()
        self.exit.Draw()
        self.entrydoor.Draw()
        self.exitdoor.Draw()
        
        # Only draw Chuck if visible
        if self.chuck_visible:
            self.player.Draw()

        for cl in self.cloud:
            cl.Draw()
        for ene in self.enemies:
            ene.Draw()
        for key in self.keys:
            key.Draw()
