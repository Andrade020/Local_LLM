"""
Interface gráfica Tkinter para LocalLLM
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from model_manager import ModelManager
from inferencer import LLMInferencer
from utils import hash_prompt, load_cache, save_cache


class LocalLLMApp:
    """Aplicação principal com interface Tkinter."""
    
    def __init__(self, initial_model_path: str = ""):
        self.root = tk.Tk()
        self.root.title("LocalLLM - Executando LLM Localmente")
        self.root.geometry("1000x700")
        
        # Estado da aplicação
        self.model_manager: Optional[ModelManager] = None
        self.inferencer: Optional[LLMInferencer] = None
        self.is_generating = False
        self.stop_generation = False
        self.current_conversation = []
        self.initial_model_path = initial_model_path
        
        # Configurações padrão
        self.settings = {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 512,
            'repeat_penalty': 1.1,
            'stop_tokens': ['</s>', '\n\n\n'],
            'use_cache': True,
        }
        
        self._setup_ui()
        self._setup_menu()
        
        # Se model_path foi fornecido, tentar carregar
        if initial_model_path:
            self.root.after(500, lambda: self._load_model(initial_model_path))
    
    def _setup_ui(self):
        """Configura todos os elementos da interface."""
        # Frame principal com divisão horizontal
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === PAINEL ESQUERDO: Histórico ===
        left_frame = ttk.Frame(main_paned, width=200)
        main_paned.add(left_frame)
        
        ttk.Label(left_frame, text="Histórico de Conversas", 
                 font=('Arial', 10, 'bold')).pack(pady=5)
        
        # Lista de conversas
        self.history_listbox = tk.Listbox(left_frame, height=20)
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.history_listbox.bind('<<ListboxSelect>>', self._load_conversation)
        
        # Botões de gerenciamento
        hist_buttons = ttk.Frame(left_frame)
        hist_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(hist_buttons, text="Nova", 
                  command=self._new_conversation).pack(side=tk.LEFT, padx=2)
        ttk.Button(hist_buttons, text="Deletar", 
                  command=self._delete_conversation).pack(side=tk.LEFT, padx=2)
        
        # === PAINEL DIREITO: Chat ===
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame)
        
        # Informações do modelo
        info_frame = ttk.LabelFrame(right_frame, text="Modelo", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.model_info_label = ttk.Label(info_frame, 
                                          text="Nenhum modelo carregado",
                                          foreground="red")
        self.model_info_label.pack(side=tk.LEFT)
        
        ttk.Button(info_frame, text="Carregar Modelo", 
                  command=self._select_and_load_model).pack(side=tk.RIGHT)
        
        # Área de saída (resposta do modelo)
        output_frame = ttk.LabelFrame(right_frame, text="Conversa", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=('Consolas', 10),
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configurar tags de formatação
        self.output_text.tag_config('user', foreground='blue', font=('Consolas', 10, 'bold'))
        self.output_text.tag_config('assistant', foreground='green')
        self.output_text.tag_config('system', foreground='gray', font=('Consolas', 9, 'italic'))
        
        # Área de entrada (prompt do usuário)
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Seu prompt:").pack(anchor=tk.W)
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=80,
            height=5,
            font=('Arial', 10)
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.input_text.bind('<Control-Return>', lambda e: self._send_message())
        
        # Botões de controle
        control_frame = ttk.Frame(input_frame)
        control_frame.pack(fill=tk.X)
        
        self.send_button = ttk.Button(
            control_frame,
            text="Enviar (Ctrl+Enter)",
            command=self._send_message
        )
        self.send_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="Parar",
            command=self._stop_generation,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Limpar",
            command=self._clear_output
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Configurações",
            command=self._open_settings
        ).pack(side=tk.RIGHT, padx=5)
        
        # Barra de progresso
        self.progress = ttk.Progressbar(right_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_label = ttk.Label(
            right_frame,
            text="Pronto. Carregue um modelo para começar.",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, padx=5)
    
    def _setup_menu(self):
        """Configura o menu da aplicação."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menu Arquivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Carregar Modelo...", 
                             command=self._select_and_load_model)
        file_menu.add_command(label="Configurações...", 
                             command=self._open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.root.quit)
        
        # Menu Ferramentas
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ferramentas", menu=tools_menu)
        tools_menu.add_command(label="Exportar Conversa...", 
                              command=self._export_conversation)
        tools_menu.add_command(label="Limpar Cache", 
                              command=self._clear_cache)
        
        # Menu Ajuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ajuda", menu=help_menu)
        help_menu.add_command(label="Sobre", command=self._show_about)
        help_menu.add_command(label="README", command=self._show_readme)
    
    def _select_and_load_model(self):
        """Abre diálogo para selecionar e carregar modelo."""
        filetypes = (
            ('Modelos GGUF', '*.gguf'),
            ('Modelos GGML', '*.bin'),
            ('Todos os arquivos', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Selecione o arquivo do modelo',
            filetypes=filetypes
        )
        
        if filename:
            self._load_model(filename)
    
    def _load_model(self, model_path: str):
        """Carrega o modelo em thread separada."""
        if self.is_generating:
            messagebox.showwarning(
                "Ocupado",
                "Aguarde a geração atual terminar."
            )
            return
        
        def load_thread():
            try:
                self._update_status("Carregando modelo...")
                self.progress.start()
                
                # Criar ModelManager
                self.model_manager = ModelManager(model_path)
                
                # Validar modelo
                is_valid, message = self.model_manager.validate_model()
                if not is_valid:
                    raise ValueError(message)
                
                # Verificar memória
                ram_ok, ram_message = self.model_manager.check_memory_requirements()
                if not ram_ok:
                    if not messagebox.askyesno(
                        "Aviso de Memória",
                        f"{ram_message}\n\nDeseja continuar mesmo assim?"
                    ):
                        raise ValueError("Carregamento cancelado pelo usuário")
                
                # Criar inferencer e carregar modelo
                self.inferencer = LLMInferencer(self.model_manager)
                self.inferencer.load_model()
                
                # Atualizar UI
                self.root.after(0, lambda: self._on_model_loaded(model_path))
                
            except Exception as e:
                logging.error(f"Erro ao carregar modelo: {e}")
                self.root.after(0, lambda: self._on_model_load_error(str(e)))
            finally:
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def _on_model_loaded(self, model_path: str):
        """Callback quando modelo é carregado com sucesso."""
        model_name = Path(model_path).name
        model_info = self.model_manager.get_model_info()
        
        info_text = (
            f"✓ {model_name} "
            f"({model_info['size_mb']:.0f} MB, "
            f"~{model_info['estimated_ram_mb']:.0f} MB RAM)"
        )
        
        self.model_info_label.config(text=info_text, foreground="green")
        self._update_status("Modelo carregado. Pronto para conversar!")
        self.send_button.config(state=tk.NORMAL)
        
        messagebox.showinfo(
            "Sucesso",
            f"Modelo carregado com sucesso!\n\n{info_text}"
        )
    
    def _on_model_load_error(self, error_message: str):
        """Callback quando há erro no carregamento."""
        self.model_info_label.config(
            text="Erro ao carregar modelo",
            foreground="red"
        )
        self._update_status("Erro no carregamento do modelo")
        
        messagebox.showerror(
            "Erro",
            f"Não foi possível carregar o modelo:\n\n{error_message}"
        )
    
    def _send_message(self):
        """Envia mensagem do usuário para o modelo."""
        if not self.inferencer or not self.inferencer.is_loaded:
            messagebox.showwarning(
                "Modelo não carregado",
                "Por favor, carregue um modelo antes de enviar mensagens."
            )
            return
        
        if self.is_generating:
            return
        
        user_input = self.input_text.get('1.0', tk.END).strip()
        if not user_input:
            return
        
        # Adicionar mensagem do usuário à conversa
        self.current_conversation.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mostrar na UI
        self._append_to_output(f"\n{'='*50}\n", 'system')
        self._append_to_output("Você: ", 'user')
        self._append_to_output(f"{user_input}\n\n", 'user')
        self._append_to_output("Assistente: ", 'assistant')
        
        # Limpar entrada
        self.input_text.delete('1.0', tk.END)
        
        # Desabilitar envio, habilitar parar
        self.send_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_generating = True
        self.stop_generation = False
        
        # Verificar cache
        cache_key = hash_prompt(user_input, self.settings)
        cached_response = None
        
        if self.settings['use_cache']:
            cached_response = load_cache(cache_key)
            if cached_response:
                logging.info("Resposta encontrada no cache")
        
        # Gerar resposta em thread separada
        def generate_thread():
            try:
                if cached_response:
                    # Usar resposta do cache
                    response = cached_response
                    self.root.after(0, lambda: self._append_to_output(response, 'assistant'))
                else:
                    # Gerar nova resposta
                    response = ""
                    for token in self.inferencer.generate_stream(
                        prompt=user_input,
                        temperature=self.settings['temperature'],
                        top_p=self.settings['top_p'],
                        max_tokens=self.settings['max_tokens'],
                        repeat_penalty=self.settings['repeat_penalty'],
                        stop=self.settings['stop_tokens']
                    ):
                        if self.stop_generation:
                            break
                        
                        response += token
                        self.root.after(0, lambda t=token: self._append_to_output(t, 'assistant'))
                    
                    # Salvar no cache
                    if self.settings['use_cache'] and not self.stop_generation:
                        save_cache(cache_key, response)
                
                # Adicionar resposta à conversa
                self.current_conversation.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat(),
                    'from_cache': cached_response is not None
                })
                
                if self.stop_generation:
                    self.root.after(0, lambda: self._append_to_output("\n\n[Geração interrompida]\n", 'system'))
                else:
                    self.root.after(0, lambda: self._append_to_output("\n", 'assistant'))
                
            except Exception as e:
                logging.error(f"Erro na geração: {e}")
                self.root.after(0, lambda: self._append_to_output(
                    f"\n\n[ERRO: {str(e)}]\n", 'system'
                ))
            finally:
                self.root.after(0, self._on_generation_complete)
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def _stop_generation(self):
        """Interrompe a geração atual."""
        self.stop_generation = True
        self._update_status("Interrompendo geração...")
    
    def _on_generation_complete(self):
        """Callback quando geração é concluída."""
        self.is_generating = False
        self.send_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._update_status("Pronto")
    
    def _append_to_output(self, text: str, tag: str = None):
        """Adiciona texto à área de saída."""
        self.output_text.config(state=tk.NORMAL)
        if tag:
            self.output_text.insert(tk.END, text, tag)
        else:
            self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def _clear_output(self):
        """Limpa a área de saída."""
        if messagebox.askyesno("Confirmar", "Limpar conversa atual?"):
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete('1.0', tk.END)
            self.output_text.config(state=tk.DISABLED)
            self.current_conversation = []
    
    def _update_status(self, message: str):
        """Atualiza a barra de status."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def _open_settings(self):
        """Abre janela de configurações."""
        SettingsWindow(self.root, self.settings)
    
    def _export_conversation(self):
        """Exporta conversa atual para arquivo."""
        if not self.current_conversation:
            messagebox.showinfo("Info", "Nenhuma conversa para exportar.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Texto", "*.txt"),
                ("JSON", "*.json"),
                ("Markdown", "*.md")
            ]
        )
        
        if filename:
            try:
                ext = Path(filename).suffix
                
                with open(filename, 'w', encoding='utf-8') as f:
                    if ext == '.json':
                        json.dump(self.current_conversation, f, indent=2, ensure_ascii=False)
                    elif ext == '.md':
                        f.write("# Conversa LocalLLM\n\n")
                        for msg in self.current_conversation:
                            role = "**Você**" if msg['role'] == 'user' else "**Assistente**"
                            f.write(f"{role}: {msg['content']}\n\n")
                    else:  # txt
                        for msg in self.current_conversation:
                            role = "Você" if msg['role'] == 'user' else "Assistente"
                            f.write(f"{role}: {msg['content']}\n\n{'='*50}\n\n")
                
                messagebox.showinfo("Sucesso", f"Conversa exportada para:\n{filename}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao exportar: {e}")
    
    def _clear_cache(self):
        """Limpa o cache de respostas."""
        if messagebox.askyesno("Confirmar", "Limpar todo o cache?"):
            try:
                cache_dir = Path('cache')
                count = 0
                for cache_file in cache_dir.glob('*.json'):
                    cache_file.unlink()
                    count += 1
                messagebox.showinfo("Sucesso", f"{count} arquivo(s) de cache removido(s).")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao limpar cache: {e}")
    
    def _new_conversation(self):
        """Inicia nova conversa."""
        if self.current_conversation:
            if messagebox.askyesno("Confirmar", "Iniciar nova conversa? A atual será perdida se não for salva."):
                self._clear_output()
        else:
            self._clear_output()
    
    def _delete_conversation(self):
        """Deleta conversa selecionada do histórico."""
        # Placeholder para funcionalidade futura
        messagebox.showinfo("Info", "Funcionalidade em desenvolvimento")
    
    def _load_conversation(self, event):
        """Carrega conversa do histórico."""
        # Placeholder para funcionalidade futura
        pass
    
    def _show_about(self):
        """Mostra informações sobre o aplicativo."""
        about_text = """
LocalLLM Desktop Application
Versão 1.0.0

Executando Large Language Models localmente
com privacidade total.

Desenvolvido com:
- Python 3.10+
- llama-cpp-python
- Tkinter

© 2024 - Software de código aberto
        """
        messagebox.showinfo("Sobre", about_text.strip())
    
    def _show_readme(self):
        """Mostra o README."""
        readme_path = Path('README.md')
        if readme_path.exists():
            try:
                import webbrowser
                webbrowser.open(str(readme_path.absolute()))
            except:
                messagebox.showinfo("README", "Abra o arquivo README.md na raiz do projeto.")
        else:
            messagebox.showinfo("README", "Arquivo README.md não encontrado.")
    
    def run(self):
        """Inicia o loop principal da aplicação."""
        self.root.mainloop()


class SettingsWindow:
    """Janela de configurações."""
    
    def __init__(self, parent, settings: dict):
        self.settings = settings
        self.window = tk.Toplevel(parent)
        self.window.title("Configurações")
        self.window.geometry("500x400")
        self.window.transient(parent)
        self.window.grab_set()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura UI da janela de configurações."""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === TAB: Inferência ===
        inference_frame = ttk.Frame(notebook, padding=10)
        notebook.add(inference_frame, text="Inferência")
        
        # Temperature
        ttk.Label(inference_frame, text="Temperature (0.0 - 2.0):").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.temp_var = tk.DoubleVar(value=self.settings['temperature'])
        temp_scale = ttk.Scale(
            inference_frame,
            from_=0.0,
            to=2.0,
            variable=self.temp_var,
            orient=tk.HORIZONTAL
        )
        temp_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)
        temp_label = ttk.Label(inference_frame, textvariable=self.temp_var)
        temp_label.grid(row=0, column=2, pady=5)
        
        # Top P
        ttk.Label(inference_frame, text="Top P (0.0 - 1.0):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.top_p_var = tk.DoubleVar(value=self.settings['top_p'])
        top_p_scale = ttk.Scale(
            inference_frame,
            from_=0.0,
            to=1.0,
            variable=self.top_p_var,
            orient=tk.HORIZONTAL
        )
        top_p_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5)
        top_p_label = ttk.Label(inference_frame, textvariable=self.top_p_var)
        top_p_label.grid(row=1, column=2, pady=5)
        
        # Max Tokens
        ttk.Label(inference_frame, text="Max Tokens:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.max_tokens_var = tk.IntVar(value=self.settings['max_tokens'])
        max_tokens_spin = ttk.Spinbox(
            inference_frame,
            from_=50,
            to=4096,
            textvariable=self.max_tokens_var,
            width=10
        )
        max_tokens_spin.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Repeat Penalty
        ttk.Label(inference_frame, text="Repeat Penalty (1.0 - 2.0):").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.repeat_penalty_var = tk.DoubleVar(value=self.settings['repeat_penalty'])
        repeat_scale = ttk.Scale(
            inference_frame,
            from_=1.0,
            to=2.0,
            variable=self.repeat_penalty_var,
            orient=tk.HORIZONTAL
        )
        repeat_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=5)
        repeat_label = ttk.Label(inference_frame, textvariable=self.repeat_penalty_var)
        repeat_label.grid(row=3, column=2, pady=5)
        
        # Use Cache
        self.use_cache_var = tk.BooleanVar(value=self.settings['use_cache'])
        ttk.Checkbutton(
            inference_frame,
            text="Usar cache de respostas",
            variable=self.use_cache_var
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        inference_frame.columnconfigure(1, weight=1)
        
        # === TAB: Avançado ===
        advanced_frame = ttk.Frame(notebook, padding=10)
        notebook.add(advanced_frame, text="Avançado")
        
        ttk.Label(advanced_frame, text="Stop Tokens (um por linha):").pack(anchor=tk.W, pady=5)
        
        self.stop_tokens_text = scrolledtext.ScrolledText(
            advanced_frame,
            width=40,
            height=10
        )
        self.stop_tokens_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.stop_tokens_text.insert('1.0', '\n'.join(self.settings['stop_tokens']))
        
        # === Botões ===
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text="Salvar",
            command=self._save_settings
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancelar",
            command=self.window.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Restaurar Padrões",
            command=self._restore_defaults
        ).pack(side=tk.LEFT, padx=5)
    
    def _save_settings(self):
        """Salva as configurações."""
        self.settings['temperature'] = self.temp_var.get()
        self.settings['top_p'] = self.top_p_var.get()
        self.settings['max_tokens'] = self.max_tokens_var.get()
        self.settings['repeat_penalty'] = self.repeat_penalty_var.get()
        self.settings['use_cache'] = self.use_cache_var.get()
        
        # Parse stop tokens
        stop_text = self.stop_tokens_text.get('1.0', tk.END).strip()
        self.settings['stop_tokens'] = [
            line.strip() for line in stop_text.split('\n') if line.strip()
        ]
        
        messagebox.showinfo("Sucesso", "Configurações salvas!")
        self.window.destroy()
    
    def _restore_defaults(self):
        """Restaura configurações padrão."""
        if messagebox.askyesno("Confirmar", "Restaurar configurações padrão?"):
            self.temp_var.set(0.7)
            self.top_p_var.set(0.9)
            self.max_tokens_var.set(512)
            self.repeat_penalty_var.set(1.1)
            self.use_cache_var.set(True)
            self.stop_tokens_text.delete('1.0', tk.END)
            self.stop_tokens_text.insert('1.0', '</s>\n\n\n')