def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Initialize ChatOpenAI with a fallback-capable free model chain."""
    settings = get_settings()
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("Missing LLM API key. Provide OPENROUTER_API_KEY.")

    # List of stable free models on OpenRouter to cycle through on failure
    free_models = [
        "openrouter/free",
        "nvidia/nemotron-3.5-lightning:free",
        "deepseek/deepseek-r1:free",
    ]
    
    # Use the primary configured model first, or default to the free rotation list
    model_to_use = settings.LLM_MODEL if settings.LLM_MODEL and "free" in settings.LLM_MODEL else free_models[0]

    return ChatOpenAI(
        model=model_to_use,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        max_tokens=512,  # Keep token output low so free providers don't throttle/timeout
        request_timeout=20.0,  # Fail fast on a dead node so it instantly tries the next fallback
        max_retries=2,  # Automatically retry across available free nodes
    )
