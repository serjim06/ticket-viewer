import gc
import signal

import torch


class TrainingPause:
    """
    Permite pausar el entrenamiento con Ctrl+C (compartido por ambos modelos).
    Ctrl+C solo marca la petición: el bucle de entrenamiento llama a pause() al terminar el batch en curso.
    Mientras está en pausa, el modelo y el estado del optimizador se mueven a la CPU y se vacía la caché de CUDA,
    de modo que la GPU queda libre salvo el contexto de CUDA del proceso.
    Desde la pausa se puede continuar (vuelve todo a la GPU) o terminar el entrenamiento.
    """

    def __init__(self, model, optimiser, device):
        self.model = model
        self.optimiser = optimiser
        self.device = torch.device(device)
        self.requested = False
        self._previous_handler = signal.signal(signal.SIGINT, self._on_sigint)
        print("Pulsa Ctrl+C para pausar el entrenamiento.")

    def _on_sigint(self, signum, frame):
        if not self.requested:
            print("\nPausa solicitada: se pausará al terminar el batch actual...")
        self.requested = True

    def close(self):
        """Restaura el manejador de Ctrl+C anterior."""
        signal.signal(signal.SIGINT, self._previous_handler)

    def _offload(self):
        """
        Mueve el modelo y los tensores del optimizador a la CPU.
        Devuelve los (estado, clave) movidos para devolver a la GPU solo esos: Adam guarda `step` en CPU
        y no admite tenerlo en CUDA.
        """
        self.model.to('cpu')
        moved = []
        for state in self.optimiser.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.device.type == 'cuda':
                    state[k] = v.to('cpu')
                    moved.append((state, k))
        gc.collect()
        torch.cuda.empty_cache()
        return moved

    def _reload(self, moved):
        self.model.to(self.device)
        for state, k in moved:
            state[k] = state[k].to(self.device)

    def _ask(self):
        # Durante el menú, Ctrl+C no debe abortar el proceso ni perder los resultados
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            while True:
                try:
                    choice = input("[c] continuar | [t] terminar y guardar resultados: ").strip().lower()
                except KeyboardInterrupt:
                    print()
                    continue
                except EOFError:
                    return 't'
                if choice in ('c', 't'):
                    return choice
        finally:
            signal.signal(signal.SIGINT, self._on_sigint)

    def pause(self):
        """
        Pausa el entrenamiento liberando la GPU y pregunta si continuar o terminar.
        Antes de llamarla hay que soltar las referencias a los tensores del batch para que se puedan liberar.
        Devuelve True si hay que terminar el entrenamiento.
        """
        on_cuda = self.device.type == 'cuda'
        before = torch.cuda.memory_reserved(self.device) if on_cuda else 0
        moved = self._offload() if on_cuda else []
        if on_cuda:
            after = torch.cuda.memory_reserved(self.device)
            print(f"Entrenamiento en pausa. GPU liberada: {(before - after) / 2**20:.0f} MB "
                  f"(quedan {after / 2**20:.0f} MB reservados por PyTorch).")
        else:
            print("Entrenamiento en pausa.")

        choice = self._ask()
        self.requested = False

        if choice == 't':
            print("Terminando el entrenamiento...")
            return True

        if on_cuda:
            self._reload(moved)
        print("Reanudando el entrenamiento...")
        return False
