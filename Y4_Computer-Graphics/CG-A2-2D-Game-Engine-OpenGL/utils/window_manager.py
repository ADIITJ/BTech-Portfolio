import glfw
from OpenGL.GL import *

class Window:
    def __init__(self, height, width):

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.windowHeight = height
        self.windowWidth = width

        self.window = glfw.create_window(
            width, height, "M24CSA007 AAP's Platform Game", None, None
        )

        # ✅ check before using the window handle
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window can't be created")

        glfw.set_window_pos(self.window, 450, 30)
        glfw.make_context_current(self.window)

        # ✅ REQUIRED FIX for core profile (prevents: No VAO bound)
        self.globalVAO = glGenVertexArrays(1)
        glBindVertexArray(self.globalVAO)

        # OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glViewport(0, 0, self.windowWidth, self.windowHeight)

        self.prevTime = glfw.get_time()

    def Close(self):
        glfw.terminate()

    def IsOpen(self):
        return not glfw.window_should_close(self.window)

    def StartFrame(self, c0, c1, c2, c3):
        currentTime = glfw.get_time()
        deltaTime = currentTime - self.prevTime
        self.prevTime = currentTime
        time = {"currentTime": currentTime, "deltaTime": deltaTime}

        glfw.poll_events()

        # optional: allow ESC to close
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

        inputs = []
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            inputs.append("1")
        if glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            inputs.append("2")
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            inputs.append("W")
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            inputs.append("A")
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            inputs.append("S")
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            inputs.append("D")
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            inputs.append("SPACE")
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            inputs.append("UP")
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            inputs.append("DOWN")
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            inputs.append("LEFT")
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            inputs.append("RIGHT")

        glClearColor(c0, c1, c2, c3)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        return inputs, time

    def EndFrame(self):
        glfw.swap_buffers(self.window)
