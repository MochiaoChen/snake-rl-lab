"""A calm, playable Snake window with human / DQN switching."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Override the active model")
    parser.add_argument("--size", type=int)
    parser.add_argument("--ai", action="store_true")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--screenshot", help="Render one frame without opening a window")
    args = parser.parse_args()
    if not 1 <= args.fps <= 20:
        parser.error("fps must be between 1 and 20")
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame
    from snake_env import SnakeEnv
    active_path = Path("active_model.json")
    active = json.loads(active_path.read_text()) if active_path.exists() else {}
    model_path = Path(args.model or active.get("model", "runs/first/best.zip"))
    config_path = model_path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    global_version = config.get("algorithm") == "GlobalDQN"
    if global_version:
        from global_env import GlobalSnakeEnv
        from global_dqn import GlobalDQN
        Env, Loader = GlobalSnakeEnv, GlobalDQN
    else:
        from stable_baselines3 import DQN
        Env, Loader = SnakeEnv, DQN
    model = None
    if model_path.exists():
        import torch
        torch.set_num_threads(2)
        model = Loader.load(model_path, device="cpu")
    trained_size = config.get("size", 10)
    env = Env(args.size or trained_size)
    obs, info = env.reset(seed=2026)
    ai = args.ai and model is not None
    if args.ai and model is None:
        parser.error(f"No model at {model_path}; run train.py first")
    pygame.init()
    width, height = 940, 716
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("贪吃蛇 · Snake Lab")
    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    fonts = {s: pygame.font.Font(font_path if Path(font_path).exists() else None, s)
             for s in (13, 15, 17, 20, 28, 34, 44)}
    color = {"bg": (246, 247, 241), "ink": (36, 59, 49), "muted": (117, 130, 119),
             "green": (51, 115, 82), "mint": (229, 239, 226), "white": (255, 255, 251),
             "line": (228, 233, 224), "red": (219, 109, 88)}
    clock = pygame.time.Clock()
    paused, started, over, running = True, False, False, True
    speed = args.fps
    best_score = 0
    next_move = 0
    desired = None
    cell = 500 / env.size
    board_x, board_y = 46, 146
    buttons = {"start": pygame.Rect(594, 480, 300, 46),
               "restart": pygame.Rect(594, 536, 144, 42),
               "ai": pygame.Rect(750, 536, 144, 42),
               "minus": pygame.Rect(594, 378, 40, 34),
               "plus": pygame.Rect(854, 378, 40, 34)}

    def text(content, x, y, size=17, tint="ink", centered=False):
        surface = fonts[size].render(content, True, color.get(tint, tint))
        if centered:
            x -= surface.get_width() / 2
        screen.blit(surface, (x, y))

    def card(rect):
        pygame.draw.rect(screen, (232, 236, 227), rect.move(0, 3), border_radius=18)
        pygame.draw.rect(screen, color["white"], rect, border_radius=18)

    def restart():
        nonlocal obs, info, paused, started, over, desired
        obs, info = env.reset()
        paused, started, over, desired = True, False, False, None

    def start_pause():
        nonlocal paused, started, next_move
        if over:
            restart()
        paused = not paused
        started = True
        next_move = pygame.time.get_ticks() + 1000 // speed

    def switch_ai():
        nonlocal ai, desired, model
        if model is None and model_path.exists():
            import torch
            torch.set_num_threads(2)
            model = Loader.load(model_path, device="cpu")
        if model is not None:
            ai = not ai
            desired = None

    def draw():
        screen.fill(color["bg"])
        text("S N A K E   L A B  /  0 1", 34, 25, 13, "muted")
        text("贪吃蛇", 32, 47, 34)
        text("慢慢来，下一口就在前面。", 35, 96, 15, "muted")
        pygame.draw.rect(screen, color["mint"], (778, 55, 114, 32), border_radius=16)
        text("AI 演示" if ai else "自由游玩", 835, 61, 15, "green", True)
        card(pygame.Rect(32, 132, 528, 528))
        for y in range(env.size):
            for x in range(env.size):
                rect = pygame.Rect(round(board_x + x * cell), round(board_y + y * cell),
                                   round(cell + 1), round(cell + 1))
                pygame.draw.rect(screen, (244, 247, 238) if (x + y) % 2 == 0 else (238, 243, 232), rect)
        if env.food is not None:
            x, y = env.food
            center = (round(board_x + (x + .5) * cell), round(board_y + (y + .5) * cell))
            radius = max(3, round(cell * .22))
            pygame.draw.circle(screen, color["red"], center, radius)
            pygame.draw.ellipse(screen, color["green"], (center[0] + 1, center[1] - radius - 5, 9, 5))
        for i, (x, y) in enumerate(env.snake):
            rect = pygame.Rect(round(board_x + x * cell + 4), round(board_y + y * cell + 4),
                               round(cell - 8), round(cell - 8))
            pygame.draw.rect(screen, color["green"] if i == 0 else (111, 162, 111), rect,
                             border_radius=max(3, int(cell * .22)))
            if i == 0:
                cx, cy = rect.center
                dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[env.direction]
                for side in (-1, 1):
                    eye = (round(cx + dx * cell * .18 - dy * side * cell * .11),
                           round(cy + dy * cell * .18 + dx * side * cell * .11))
                    pygame.draw.circle(screen, color["white"], eye, max(2, int(cell * .065)))
        card(pygame.Rect(594, 132, 300, 150))
        text("本局得分", 615, 152, 15, "muted")
        text(str(info["score"]).zfill(2), 612, 172, 44)
        pygame.draw.line(screen, color["line"], (615, 236), (872, 236))
        text(f"最高  {best_score}       蛇长  {len(env.snake)}", 615, 249, 15, "muted")
        text("节奏", 594, 315, 20)
        text("方向键转向，反向掉头无效", 594, 346, 15, "muted")
        text(f"{speed} 格 / 秒", 744, 383, 17, centered=True)
        for name, label in (("minus", "−"), ("plus", "+")):
            pygame.draw.rect(screen, color["mint"], buttons[name], border_radius=9)
            text(label, buttons[name].centerx, buttons[name].y + 3, 20, "green", True)
        text("也可以按 − / + 调节速度", 594, 429, 13, "muted")
        pygame.draw.rect(screen, color["green"], buttons["start"], border_radius=12)
        label = "再来一局" if over else "开始游戏" if not started else "继续" if paused else "暂停"
        text(label, 744, 490, 17, "white", True)
        for name, label in (("restart", "重新开始"), ("ai", "自己玩" if ai else "AI 接管")):
            pygame.draw.rect(screen, color["mint"] if name == "restart" or model or model_path.exists() else color["line"],
                             buttons[name], border_radius=12)
            text(label, buttons[name].centerx, buttons[name].y + 10, 15,
                 "green" if name == "restart" or model or model_path.exists() else "muted", True)
        text("空格 开始 / 暂停    R 重开", 594, 602, 13, "muted")
        text("A 切换 AI    Esc 退出", 594, 625, 13, "muted")
        if model is None:
            text("模型已就绪，点击 AI 接管。" if model_path.exists() else "AI 正在准备，先自己玩一局。", 594, 668, 13, "muted")
        else:
            text("全局 RL · v2 · 已加载模型" if global_version else "局部 RL · v1 · 已加载模型", 594, 668, 13, "muted")
        text("吃到食物 +1 · 碰到边界或身体结束", 34, 677, 13, "muted")
        if paused or over:
            overlay = pygame.Surface((500, 500), pygame.SRCALPHA)
            overlay.fill((246, 247, 241, 150))
            screen.blit(overlay, (board_x, board_y))
            message = "这一局结束啦" if over else "准备好了吗？" if not started else "休息一下"
            text(message, 296, 341, 28, centered=True)
            detail = "按 R 重开，或点击「再来一局」" if over else "按方向键或空格开始" if not started else "空格继续"
            text(detail, 296, 388, 15, "muted", True)
        pygame.display.flip()

    draw()
    if args.screenshot:
        Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, args.screenshot)
        pygame.quit()
        return
    direction_keys = {pygame.K_RIGHT: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_UP: 3}
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for name, rect in buttons.items():
                    if rect.collidepoint(event.pos):
                        if name == "start":
                            if over:
                                restart()
                            start_pause()
                        elif name == "restart":
                            restart()
                        elif name == "ai":
                            switch_ai()
                        elif name == "minus":
                            speed = max(1, speed - 1)
                        elif name == "plus":
                            speed = min(20, speed + 1)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    switch_ai()
                elif event.key == pygame.K_SPACE:
                    start_pause()
                elif event.key == pygame.K_r:
                    restart()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(1, speed - 1)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(20, speed + 1)
                elif event.key in direction_keys and not ai and not over:
                    d = direction_keys[event.key]
                    if (d - env.direction) % 4 != 2:
                        desired = d
                        if not started:
                            start_pause()
        now = pygame.time.get_ticks()
        if not over and not paused and now >= next_move:
            next_move = now + 1000 // speed
            action = int(model.predict(obs, deterministic=True)[0]) if ai else (
                1 if desired is not None and (desired - env.direction) % 4 == 3 else
                2 if desired is not None and (desired - env.direction) % 4 == 1 else 0)
            desired = None
            obs, _, terminated, truncated, info = env.step(action)
            best_score = max(best_score, info["score"])
            over = terminated or truncated
        draw()
        clock.tick(60)
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
