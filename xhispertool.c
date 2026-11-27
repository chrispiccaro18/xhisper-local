/*
 * xhisper - Whisper for Linux
 * Combined daemon and client for text input via uinput
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <libgen.h>
#include <glob.h>
#include <dlfcn.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <linux/uinput.h>
#define KEY_LEFTCTRL 29
#define KEY_RIGHTCTRL 97
#define KEY_LEFTALT 56
#define KEY_RIGHTALT 100
#define KEY_LEFTSHIFT 42
#define KEY_RIGHTSHIFT 54
#define KEY_LEFTMETA 125
#define KEY_V 47
#define FLAG_UPPERCASE 0x80000000

// Function prototypes
void cleanup(void);
void emit(int type, int code, int val);
void do_paste(void);
void type_char(unsigned char c);
void do_backspace(void);
void do_key(int keycode);
int setup_uinput(void);
int setup_socket(void);
int run_daemon(void);
void show_usage(void);
int wait_keys_released(int max_ms);
int run_client(int argc, char *argv[]);

// ASCII to Linux keycode mapping for US QWERTY layout
// Independently derived from:
// - ASCII character set specification (characters 0-127)
// - Linux input-event-codes.h KEY_* constants
// - US QWERTY keyboard physical layout
// Each entry maps an ASCII code to either:
//   -1 (unmapped/unsupported)
//   KEY_* constant for unshifted characters
//   KEY_* | FLAG_UPPERCASE for shifted characters
static const int32_t ascii2keycode_map[128] = {
	// Control characters (0x00-0x1f): mostly unmapped except tab and enter
	-1,-1,-1,-1,-1,-1,-1,-1,  // 0x00-0x07
	-1,KEY_TAB,KEY_ENTER,-1,-1,-1,-1,-1,  // 0x08-0x0f (tab=0x09, enter=0x0a)
	-1,-1,-1,-1,-1,-1,-1,-1,  // 0x10-0x17
	-1,-1,-1,-1,-1,-1,-1,-1,  // 0x18-0x1f

	// Printable characters (0x20-0x7e)
	// Space and symbols (0x20-0x2f)
	KEY_SPACE,                      // 0x20 ' '
	KEY_1|FLAG_UPPERCASE,           // 0x21 '!' (shift+1)
	KEY_APOSTROPHE|FLAG_UPPERCASE,  // 0x22 '"' (shift+')
	KEY_3|FLAG_UPPERCASE,           // 0x23 '#' (shift+3)
	KEY_4|FLAG_UPPERCASE,           // 0x24 '$' (shift+4)
	KEY_5|FLAG_UPPERCASE,           // 0x25 '%' (shift+5)
	KEY_7|FLAG_UPPERCASE,           // 0x26 '&' (shift+7)
	KEY_APOSTROPHE,                 // 0x27 '''
	KEY_9|FLAG_UPPERCASE,           // 0x28 '(' (shift+9)
	KEY_0|FLAG_UPPERCASE,           // 0x29 ')' (shift+0)
	KEY_8|FLAG_UPPERCASE,           // 0x2a '*' (shift+8)
	KEY_EQUAL|FLAG_UPPERCASE,       // 0x2b '+' (shift+=)
	KEY_COMMA,                      // 0x2c ','
	KEY_MINUS,                      // 0x2d '-'
	KEY_DOT,                        // 0x2e '.'
	KEY_SLASH,                      // 0x2f '/'

	// Digits (0x30-0x39)
	KEY_0,KEY_1,KEY_2,KEY_3,KEY_4,KEY_5,KEY_6,KEY_7,KEY_8,KEY_9,

	// More symbols (0x3a-0x40)
	KEY_SEMICOLON|FLAG_UPPERCASE,   // 0x3a ':' (shift+;)
	KEY_SEMICOLON,                  // 0x3b ';'
	KEY_COMMA|FLAG_UPPERCASE,       // 0x3c '<' (shift+,)
	KEY_EQUAL,                      // 0x3d '='
	KEY_DOT|FLAG_UPPERCASE,         // 0x3e '>' (shift+.)
	KEY_SLASH|FLAG_UPPERCASE,       // 0x3f '?' (shift+/)
	KEY_2|FLAG_UPPERCASE,           // 0x40 '@' (shift+2)

	// Uppercase letters (0x41-0x5a): A-Z
	KEY_A|FLAG_UPPERCASE,KEY_B|FLAG_UPPERCASE,KEY_C|FLAG_UPPERCASE,KEY_D|FLAG_UPPERCASE,
	KEY_E|FLAG_UPPERCASE,KEY_F|FLAG_UPPERCASE,KEY_G|FLAG_UPPERCASE,KEY_H|FLAG_UPPERCASE,
	KEY_I|FLAG_UPPERCASE,KEY_J|FLAG_UPPERCASE,KEY_K|FLAG_UPPERCASE,KEY_L|FLAG_UPPERCASE,
	KEY_M|FLAG_UPPERCASE,KEY_N|FLAG_UPPERCASE,KEY_O|FLAG_UPPERCASE,KEY_P|FLAG_UPPERCASE,
	KEY_Q|FLAG_UPPERCASE,KEY_R|FLAG_UPPERCASE,KEY_S|FLAG_UPPERCASE,KEY_T|FLAG_UPPERCASE,
	KEY_U|FLAG_UPPERCASE,KEY_V|FLAG_UPPERCASE,KEY_W|FLAG_UPPERCASE,KEY_X|FLAG_UPPERCASE,
	KEY_Y|FLAG_UPPERCASE,KEY_Z|FLAG_UPPERCASE,

	// Brackets and symbols (0x5b-0x60)
	KEY_LEFTBRACE,                  // 0x5b '['
	KEY_BACKSLASH,                  // 0x5c '\'
	KEY_RIGHTBRACE,                 // 0x5d ']'
	KEY_6|FLAG_UPPERCASE,           // 0x5e '^' (shift+6)
	KEY_MINUS|FLAG_UPPERCASE,       // 0x5f '_' (shift+-)
	KEY_GRAVE,                      // 0x60 '`'

	// Lowercase letters (0x61-0x7a): a-z
	KEY_A,KEY_B,KEY_C,KEY_D,KEY_E,KEY_F,KEY_G,KEY_H,
	KEY_I,KEY_J,KEY_K,KEY_L,KEY_M,KEY_N,KEY_O,KEY_P,
	KEY_Q,KEY_R,KEY_S,KEY_T,KEY_U,KEY_V,KEY_W,KEY_X,
	KEY_Y,KEY_Z,

	// Final symbols (0x7b-0x7e)
	KEY_LEFTBRACE|FLAG_UPPERCASE,   // 0x7b '{' (shift+[)
	KEY_BACKSLASH|FLAG_UPPERCASE,   // 0x7c '|' (shift+\)
	KEY_RIGHTBRACE|FLAG_UPPERCASE,  // 0x7d '}' (shift+])
	KEY_GRAVE|FLAG_UPPERCASE,       // 0x7e '~' (shift+`)

	-1  // 0x7f DEL (unmapped)
};

static int fd_uinput = -1;
static int fd_socket = -1;

void cleanup() {
    if (fd_uinput >= 0) {
        ioctl(fd_uinput, UI_DEV_DESTROY);
        close(fd_uinput);
    }
    if (fd_socket >= 0) {
        close(fd_socket);
    }
}

void emit(int type, int code, int val) {
    struct input_event ie = {
        .type = type,
        .code = code,
        .value = val
    };
    write(fd_uinput, &ie, sizeof(ie));
}

void do_paste() {
    emit(EV_KEY, KEY_LEFTCTRL, 1);
    emit(EV_SYN, SYN_REPORT, 0);
    usleep(8000);
    emit(EV_KEY, KEY_V, 1);
    emit(EV_SYN, SYN_REPORT, 0);
    usleep(8000);
    emit(EV_KEY, KEY_V, 0);
    emit(EV_SYN, SYN_REPORT, 0);
    usleep(2000);
    emit(EV_KEY, KEY_LEFTCTRL, 0);
    emit(EV_SYN, SYN_REPORT, 0);
}

void type_char(unsigned char c) {
    if (c >= 128) return;

    int32_t kdef = ascii2keycode_map[c];
    if (kdef == -1) return;

    uint16_t keycode = kdef & 0xffff;

    if (kdef & FLAG_UPPERCASE) {
        emit(EV_KEY, KEY_LEFTSHIFT, 1);
        emit(EV_SYN, SYN_REPORT, 0);
        usleep(2000);
    }

    emit(EV_KEY, keycode, 1);
    emit(EV_SYN, SYN_REPORT, 0);
    usleep(8000);

    emit(EV_KEY, keycode, 0);
    emit(EV_SYN, SYN_REPORT, 0);
    usleep(2000);

    if (kdef & FLAG_UPPERCASE) {
        emit(EV_KEY, KEY_LEFTSHIFT, 0);
        emit(EV_SYN, SYN_REPORT, 0);
    }
}

void do_backspace() {
    emit(EV_KEY, KEY_BACKSPACE, 1);
    emit(EV_SYN, SYN_REPORT, 0);
    usleep(8000);
    emit(EV_KEY, KEY_BACKSPACE, 0);
    emit(EV_SYN, SYN_REPORT, 0);
}

void do_key(int keycode) {
    emit(EV_KEY, keycode, 1);
    emit(EV_SYN, SYN_REPORT, 0);
    usleep(8000);
    emit(EV_KEY, keycode, 0);
    emit(EV_SYN, SYN_REPORT, 0);
}

int setup_uinput() {
    fd_uinput = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd_uinput < 0) {
        perror("failed to open /dev/uinput");
        return -1;
    }

    ioctl(fd_uinput, UI_SET_EVBIT, EV_KEY);

    // Register letters
    for (int i = KEY_Q; i <= KEY_P; i++) ioctl(fd_uinput, UI_SET_KEYBIT, i);
    for (int i = KEY_A; i <= KEY_L; i++) ioctl(fd_uinput, UI_SET_KEYBIT, i);
    for (int i = KEY_Z; i <= KEY_M; i++) ioctl(fd_uinput, UI_SET_KEYBIT, i);

    // Register numbers
    for (int i = KEY_1; i <= KEY_0; i++) ioctl(fd_uinput, UI_SET_KEYBIT, i);

    // Register special keys
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_SPACE);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_MINUS);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_EQUAL);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_LEFTBRACE);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_RIGHTBRACE);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_SEMICOLON);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_APOSTROPHE);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_GRAVE);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_BACKSLASH);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_COMMA);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_DOT);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_SLASH);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_TAB);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_ENTER);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_BACKSPACE);

    // Register modifiers
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_LEFTCTRL);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_RIGHTCTRL);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_LEFTALT);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_RIGHTALT);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_LEFTSHIFT);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_RIGHTSHIFT);
    ioctl(fd_uinput, UI_SET_KEYBIT, KEY_LEFTMETA);

    struct uinput_setup setup = {0};
    setup.id.bustype = BUS_USB;
    setup.id.vendor = 0x1234;
    setup.id.product = 0x5678;
    snprintf(setup.name, UINPUT_MAX_NAME_SIZE, "xhisper");

    if (ioctl(fd_uinput, UI_DEV_SETUP, &setup) < 0) {
        perror("failed to setup uinput device");
        return -1;
    }
    if (ioctl(fd_uinput, UI_DEV_CREATE) < 0) {
        perror("failed to create uinput device");
        return -1;
    }

    usleep(100000);
    return 0;
}

int setup_socket() {
    fd_socket = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd_socket < 0) {
        perror("failed to create socket");
        return -1;
    }

    // Use abstract namespace socket (Linux-specific)
    // First byte is null, no filesystem entry, kernel manages lifecycle
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    addr.sun_path[0] = '\0';
    strncpy(addr.sun_path + 1, "xhisper_socket", sizeof(addr.sun_path) - 2);

    if (bind(fd_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        if (errno == EADDRINUSE) {
            fprintf(stderr, "xhispertoold is already running\n");
        } else {
            perror("failed to bind socket");
        }
        return -1;
    }

    return 0;
}

// Daemon mode
int run_daemon() {
    atexit(cleanup);

    if (setup_uinput() < 0) {
        return 1;
    }

    if (setup_socket() < 0) {
        return 1;
    }

    printf("xhispertoold: listening on @xhisper_socket\n");

    char buf[2];
    while (1) {
        ssize_t n = recv(fd_socket, buf, sizeof(buf), 0);
        if (n >= 1) {
            char cmd = buf[0];
            if (cmd == 'p') {
                do_paste();
            } else if (cmd == 't' && n == 2) {
                type_char((unsigned char)buf[1]);
            } else if (cmd == 'b') {
                do_backspace();
            } else if (cmd == 'r') {
                do_key(KEY_RIGHTALT);
            } else if (cmd == 'L') {
                do_key(KEY_LEFTALT);
            } else if (cmd == 'C') {
                do_key(KEY_LEFTCTRL);
            } else if (cmd == 'R') {
                do_key(KEY_RIGHTCTRL);
            } else if (cmd == 'S') {
                do_key(KEY_LEFTSHIFT);
            } else if (cmd == 'T') {
                do_key(KEY_RIGHTSHIFT);
            } else if (cmd == 'M') {
                do_key(KEY_LEFTMETA);
            }
        }
    }

    return 0;
}

// wait-keys-released: wait until no keyboard key is held
//
// xhisper runs from a keyboard shortcut whose keys are usually still down
// when it starts. Keys typed meanwhile arrive combined with the shortcut's
// modifiers (Alt+Backspace deletes a word in many apps), and on X11 they
// don't reach the app at all while the shortcut's key grab lasts. Reading
// the actual key state lets xhisper go ahead as soon as the keys are up,
// however long they are held. Sources, in order:
//   1. the kernel's state of every keyboard (EVIOCGKEY), X11 and Wayland
//      alike. Used only if every /dev/input/event* opens (the input group):
//      with some unreadable, a held key could go unseen.
//   2. the X server's (XQueryKeymap), on X11 only: under Wayland, Xwayland
//      only sees keys sent to X windows. libX11 is loaded at run time, so
//      building needs no X headers and Wayland-only systems need no libX11.
// Exit 0 once no key is held (stdout: "<source> <ms waited>"), 3 if a key is
// still held after max_ms (stdout: "<source> timeout"; stderr: which key),
// 4 if neither source is available. These differ from the other commands'
// 1 and 2, so a script running an older binary sees neither "released" nor
// "held".

#define KEYS_EXIT_RELEASED 0
#define KEYS_EXIT_HELD 3
#define KEYS_EXIT_UNAVAILABLE 4
#define KEYS_POLL_US 5000
// After the kernel shows the release, give the display server a moment to
// process the same event before anything is typed (XQueryKeymap already is
// the server's own state)
#define KEYS_EVDEV_SETTLE_US 20000
#define MAX_KEYBOARDS 64

#define BITS_PER_LONG (sizeof(long) * 8)
#define NLONGS(n) (((n) + BITS_PER_LONG - 1) / BITS_PER_LONG)

typedef struct {
    int fd;
    char name[128];
} keyboard_t;

static long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
}

// First keyboard key set in an EV_KEY bitmap, or 0. Codes from BTN_MISC up
// are mouse, joystick and other buttons, not keys.
static int first_key(const unsigned long *bits) {
    for (int code = 1; code < BTN_MISC; code++) {
        if ((bits[code / BITS_PER_LONG] >> (code % BITS_PER_LONG)) & 1) return code;
    }
    return 0;
}

// Open every input device that has keyboard keys, except our own virtual
// keyboard. Returns how many, or -1 if any device could not be opened.
static int open_keyboards(keyboard_t *kbds, int max) {
    glob_t g;
    int n = 0, unreadable = 0;
    if (glob("/dev/input/event*", 0, NULL, &g) != 0) return -1;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        int fd = open(g.gl_pathv[i], O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) {
            unreadable = 1;
            break;
        }
        char name[128] = "";
        unsigned long caps[NLONGS(KEY_MAX + 1)] = {0};
        ioctl(fd, EVIOCGNAME(sizeof(name) - 1), name);
        if (n == max || strcmp(name, "xhisper") == 0 ||  // see setup_uinput
            ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(caps)), caps) < 0 || !first_key(caps)) {
            close(fd);
            continue;
        }
        kbds[n].fd = fd;
        snprintf(kbds[n].name, sizeof(kbds[n].name), "%s", name);
        n++;
    }
    globfree(&g);
    if (unreadable) {
        for (int i = 0; i < n; i++) close(kbds[i].fd);
        return -1;
    }
    return n;
}

static int wait_keys_evdev(keyboard_t *kbds, int n, int max_ms) {
    long long start = now_ms();
    int was_held = 0;
    for (;;) {
        int held = 0;
        const char *device = "";
        for (int i = 0; i < n && !held; i++) {
            unsigned long state[NLONGS(KEY_MAX + 1)] = {0};
            if (ioctl(kbds[i].fd, EVIOCGKEY(sizeof(state)), state) < 0) continue;  // unplugged
            held = first_key(state);
            device = kbds[i].name;
        }
        long long waited = now_ms() - start;
        if (!held) {
            if (was_held) usleep(KEYS_EVDEV_SETTLE_US);
            printf("evdev %lld\n", waited);
            return KEYS_EXIT_RELEASED;
        }
        if (waited >= max_ms) {
            printf("evdev timeout\n");
            fprintf(stderr, "wait-keys-released: key %d on \"%s\" still held after %dms\n",
                    held, device, max_ms);
            return KEYS_EXIT_HELD;
        }
        was_held = 1;
        usleep(KEYS_POLL_US);
    }
}

static int wait_keys_x11(int max_ms) {
    const char *session = getenv("XDG_SESSION_TYPE");
    if (!getenv("DISPLAY") || getenv("WAYLAND_DISPLAY") ||
        (session && strcmp(session, "wayland") == 0)) {
        return KEYS_EXIT_UNAVAILABLE;
    }
    void *lib = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) return KEYS_EXIT_UNAVAILABLE;
    void *(*open_display)(const char *);
    int (*query_keymap)(void *, char *);
    int (*close_display)(void *);
    *(void **)&open_display = dlsym(lib, "XOpenDisplay");
    *(void **)&query_keymap = dlsym(lib, "XQueryKeymap");
    *(void **)&close_display = dlsym(lib, "XCloseDisplay");
    void *display = (open_display && query_keymap && close_display) ? open_display(NULL) : NULL;
    if (!display) {
        dlclose(lib);
        return KEYS_EXIT_UNAVAILABLE;
    }

    long long start = now_ms();
    int rc;
    for (;;) {
        unsigned char keys[32];
        int held = 0;
        query_keymap(display, (char *)keys);
        for (int keycode = 8; keycode < 256 && !held; keycode++) {
            if (keys[keycode / 8] & (1 << (keycode % 8))) held = keycode;
        }
        long long waited = now_ms() - start;
        if (!held) {
            printf("x11 %lld\n", waited);
            rc = KEYS_EXIT_RELEASED;
            break;
        }
        if (waited >= max_ms) {
            printf("x11 timeout\n");
            fprintf(stderr, "wait-keys-released: X keycode %d still held after %dms\n",
                    held, max_ms);
            rc = KEYS_EXIT_HELD;
            break;
        }
        usleep(KEYS_POLL_US);
    }
    close_display(display);
    dlclose(lib);
    return rc;
}

int wait_keys_released(int max_ms) {
    keyboard_t kbds[MAX_KEYBOARDS];
    int n = open_keyboards(kbds, MAX_KEYBOARDS);
    if (n > 0) {
        int rc = wait_keys_evdev(kbds, n, max_ms);
        for (int i = 0; i < n; i++) close(kbds[i].fd);
        return rc;
    }
    int rc = wait_keys_x11(max_ms);
    if (rc == KEYS_EXIT_UNAVAILABLE) {
        fprintf(stderr, "wait-keys-released: no key state available: /dev/input/event* "
                        "not all readable (input group) and no X11 session\n");
    }
    return rc;
}

// Client mode
void show_usage() {
    fprintf(stderr,
        "Usage:\n"
        "  xhispertool paste            - Paste from clipboard (Ctrl+V)\n"
        "  xhispertool type <char>      - Type a single ASCII character\n"
        "  xhispertool backspace        - Press backspace\n"
        "  xhispertool wait-keys-released [MAX_MS]\n"
        "                               - Wait until no key is held (default 2000ms)\n"
        "\n"
        "Input switching keys:\n"
        "  xhispertool leftalt          - Press left alt\n"
        "  xhispertool rightalt         - Press right alt\n"
        "  xhispertool leftctrl         - Press left ctrl\n"
        "  xhispertool rightctrl        - Press right ctrl\n"
        "  xhispertool leftshift        - Press left shift\n"
        "  xhispertool rightshift       - Press right shift\n"
        "  xhispertool super            - Press super (Windows key)\n"
        "\n"
        "Daemon:\n"
        "  xhispertoold                 - Run daemon (or xhispertool --daemon)\n"
    );
}

int run_client(int argc, char *argv[]) {
    if (argc < 2) {
        show_usage();
        return 1;
    }

    // Reads key state itself; needs no daemon
    if (strcmp(argv[1], "wait-keys-released") == 0) {
        int max_ms = argc > 2 ? atoi(argv[2]) : 2000;
        return wait_keys_released(max_ms > 0 ? max_ms : 2000);
    }

    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("failed to create socket");
        return 1;
    }

    // Use abstract namespace socket (same as daemon)
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    addr.sun_path[0] = '\0';
    strncpy(addr.sun_path + 1, "xhisper_socket", sizeof(addr.sun_path) - 2);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        int err = errno;
        fprintf(stderr, "failed to connect to xhispertoold: %s\n", strerror(err));

        switch (err) {
            case ENOENT:
            case ECONNREFUSED:
                fprintf(stderr, "Please check if xhispertoold is running.\n");
                fprintf(stderr, "Start it with: xhispertoold &\n");
                break;
            case EACCES:
            case EPERM:
                fprintf(stderr, "Permission denied. Check socket permissions.\n");
                break;
        }
        close(fd);
        return 2;
    }

    char buf[2];
    ssize_t len = 0;

    if (strcmp(argv[1], "paste") == 0) {
        buf[0] = 'p';
        len = 1;
    } else if (strcmp(argv[1], "backspace") == 0) {
        buf[0] = 'b';
        len = 1;
    } else if (strcmp(argv[1], "rightalt") == 0) {
        buf[0] = 'r';
        len = 1;
    } else if (strcmp(argv[1], "leftalt") == 0) {
        buf[0] = 'L';
        len = 1;
    } else if (strcmp(argv[1], "leftctrl") == 0) {
        buf[0] = 'C';
        len = 1;
    } else if (strcmp(argv[1], "rightctrl") == 0) {
        buf[0] = 'R';
        len = 1;
    } else if (strcmp(argv[1], "leftshift") == 0) {
        buf[0] = 'S';
        len = 1;
    } else if (strcmp(argv[1], "rightshift") == 0) {
        buf[0] = 'T';
        len = 1;
    } else if (strcmp(argv[1], "super") == 0) {
        buf[0] = 'M';
        len = 1;
    } else if (strcmp(argv[1], "type") == 0) {
        if (argc != 3 || strlen(argv[2]) != 1) {
            fprintf(stderr, "Error: 'type' requires exactly one character argument\n");
            show_usage();
            close(fd);
            return 1;
        }
        buf[0] = 't';
        buf[1] = argv[2][0];
        len = 2;
    } else {
        fprintf(stderr, "Error: Unknown command '%s'\n", argv[1]);
        show_usage();
        close(fd);
        return 1;
    }

    if (write(fd, buf, len) != len) {
        perror("failed to send command");
        close(fd);
        return 1;
    }

    close(fd);
    return 0;
}

int main(int argc, char *argv[]) {
    // Detect mode: daemon or client
    char *prog = basename(argv[0]);

    if (strcmp(prog, "xhispertoold") == 0 ||
        (argc > 1 && strcmp(argv[1], "--daemon") == 0)) {
        return run_daemon();
    } else {
        return run_client(argc, argv);
    }
}
